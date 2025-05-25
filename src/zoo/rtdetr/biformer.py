from collections import OrderedDict
from functools import partial
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models import register_model
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
from typing import Tuple
from torch import Tensor

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x

class KVGather(nn.Module): # 用于根据路由索引和权重从键值对张量中选择特定的键值对
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard'] # 断言权重相乘的方式必须在 ['none', 'soft', 'hard'] 中，确保输入的方式是合法的
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor): # 接受路由索引 r_idx、路由权重 r_weight 和键值对张量 kv 作为输入
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1) # 获取路由索引中的 Top-k 数量
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?

        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy kv.view(n, 1, p2, w2, c_kv)：将键值对张量 kv 重新形状为 (n, 1, p2, w2, c_kv)，在这个过程中，张量的维度被重新排列以匹配 torch.gather 函数的需求
                               dim=2, # 通过调用 expand 方法，将第二维度扩展为 p2，即将维度为 1 的维度复制 p2 次，其他维度保持不变
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv) 将路由索引张量 r_idx 重新形状为 (n, p2, topk, 1, 1)，以匹配 torch.gather 函数的索引需求
                               )# 通过调用 expand 方法，将索引张量的最后两个维度分别扩展为 w2 和 c_kv，以匹配键值对张量的形状
        if self.mul_weight == 'soft': # 如果权重相乘的方式为 'soft'，则将权重乘以选择的键值对张量
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard': # 如果权重相乘的方式为 'hard'，抛出未实现的异常，表示硬路由方式尚未实现
            raise NotImplementedError('differentiable hard routing TBA')
        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True): # 输入维度 dim、查询键维度 qk_dim 和是否包含偏置项 bias 作为参数
        super().__init__()
        self.dim = dim  # 将输入维度和查询键维度分别保存为类的属性
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias) # 创建一个线性层 qkv，输入维度为 dim，输出维度为 qk_dim + qk_dim + dim，即查询维度、键维度和值维度的总和，可以选择是否包含偏置项

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1) # 将输入张量 x 经过线性映射操作得到的结果进行分割，分别得到查询 q 和键值对 kv，分割的位置根据查询键维度和值维度进行划分
        return q, kv

class TopkRouting(nn.Module): # 用于执行可微分的 Top-k 路由操作
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key #mg
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self,
                 qk_dim, # 接受查询和键的特征维度 qk_dim
                 topk=4, # Top-k 值 topk
                 qk_scale=None, # 温度参数 qk_scale
                 param_routing=False, # 是否包含学习参数的路由单元
                 diff_routing=False): # 是否进行可微分路由 diff_routing 作为参数
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity() # 根据是否包含学习参数的路由单元，创建一个线性层或者恒等映射
        # routing activation
        self.routing_act = nn.Softmax(dim=-1) # 创建一个 Softmax 激活函数，用于计算路由概率

    def forward(self, query, key):
        if not self.diff_routing: # 如果不进行可微分路由，则对查询和键进行 detach() 操作，即断开梯度传播
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 通过线性映射层或者恒等映射，得到查询和键的映射结果
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2) 计算注意力logit，即查询和键的点积，经过缩放后，得到注意力logit
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k) 根据 Top-k 值，选择注意力logit 中的前 k 个值，并返回对应的索引。
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k) 通过 Softmax 激活函数计算路由权重，得到路由权重
        return r_weight, topk_index # 返回路由权重 r_weight 和 Top-k 索引 topk_index

class BiLevelRoutingAttention(nn.Module):
    def __init__(self, dim, # 输入特征的维度
                 num_heads=8,
                 n_win=7, # 窗口的数量，默认为 7
                 qk_dim=None, # 查询和键的维度，如果未指定，则默认为 None，在后续会根据情况设置为输入特征的维度 dim
                 qk_scale=None, # 查询和键的缩放因子，默认为 None
                 kv_per_win=4, # 每个窗口中键值对的数量，默认为 4
                 kv_downsample_ratio=4, #键值对下采样的比率，默认为 4
                 kv_downsample_kernel=None,# 键值对下采样的卷积核，默认为 None
                 kv_downsample_mode='identity', # 键值对下采样的模式，默认为 'identity'，表示不进行下采样。
                 topk=4, param_attention="qkvo", # 参数注意力的设置，默认为 "qkvo"，表示同时使用查询、键、值和输出的线性映射
                 param_routing=False,# 是否使用参数路由，默认为 False
                 diff_routing=False, # 是否使用不同的路由方式，默认为 False
                 soft_routing=False, # 是否使用软路由，默认为 False
                 side_dwconv=3, # 侧边深度可分离卷积的尺寸，默认为 3
                 auto_pad=True): # 是否自动填充，默认为 True
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5
# 这个卷积操作用于生成局部位置嵌入
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x) # 这个函数用于在不需要卷积操作时返回一个全零张量，保持数据的形状不变

        ################ global routing setting ###mg##############
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False # 断言不可以同时设置参数路由为 True 且不使用不同的路由方式，因为参数路由需要使用不同的路由方式。
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing) # 根据参数设置创建一个 TopkRouting 路由器，其中包括查询键维度、缩放因子、Top-k 值、不同的路由方式和参数路由的设置。
        if self.soft_routing: # soft routing, always diffrentiable (if no detach) # 如果设置了软路由，则设置权重乘法方式为 'soft'。
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing 如果设置了不同的路由方式，则设置权重乘法方式为 'hard'，表示硬可微分路由。
            mul_weight = 'hard'
        else:  # hard non-differentiable routing 如果没有设置软路由或不同的路由方式，则设置权重乘法方式为 'none'，表示硬非可微分路由。
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight) # 根据权重乘法方式创建一个 KVGather 模块，用于收集键值信息

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo': # 如果参数设置为 'qkvo'，则创建一个 QKVLinear 模块用于查询、键、值线性映射，同时创建一个全连接层 nn.Linear 用于最终输出
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv': # 如果参数设置为 'qkv'，同样创建一个 QKVLinear 模块用于查询、键、值线性映射，但是输出直接使用 nn.Identity()，表示不进行额外的映射处理
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win) # 创建一个自适应平均池化层 nn.AdaptiveAvgPool2d
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win) # 创建一个自适应最大池化层 nn.AdaptiveMaxPool2d
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None # 创建一个最大池化层 nn.MaxPool2d 或者恒等映射 nn.Identity()，取决于下采样比率是否大于 1
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None # 创建一个平均池化层 nn.AvgPool2d 或者恒等映射 nn.Identity()，取决于下采样比率是否大于 1。
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling 直接创建一个恒等映射 nn.Identity()，表示不进行键值对下采样操作
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1) # 用于在局部注意力机制中对输入进行 Softmax 操作。通过设置 dim=-1，表示对最后一个维度进行 Softmax 操作，通常用于计算注意力权重

        self.auto_pad=auto_pad # 传入的 auto_pad 参数赋值给类的属性 auto_pad，表示是否自动填充的设置。这个属性可能在模型的其他部分用于控制是否进行自动填充操作

    def forward(self, x):
        N, H, W, C = x.size() # 获取输入张量 x 的大小，其中 N 表示批量大小，H 和 W 表示高度和宽度，C 表示通道数

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win) # 对输入张量 x 进行重新排列，将其划分为多个小窗口，以便进行局部注意力计算
        q, kv = self.qkv(x) # 通过查询键值线性映射获取查询 q 和键值对 kv
        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c') # 对查询张量 q 进行重新排列，将其从 (n, p^2, h, w, c) 的形状转换为 (n, p^2, h*w, c) 的形状，将查询张量按照像素级重新组织
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w')) # 对键值对张量 kv 进行重新排列，并通过 kv_down 模块进行下采样操作，将其从 (n, p^2, h, w, c) 的形状转换为 (np^2, c, h, w) 的形状，
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win) # 然后再将其重新排列为 (n, ji, h*w, c) 的形状，其中 j 和 i 分别为窗口的高度和宽度
# 计算窗口级的查询 q_win 和键 k_win，分别通过对查询张量 q 在第 2 和第 3 维度上进行平均值计算，以及对键值对张量 kv 在第 2 和第 3 维度上进行平均值计算获取键
        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp 对键值对张量中的一部分（从 self.qk_dim 开始的部分）进行重新排列，并通过侧边深度可分离卷积模块 lepe 进行处理，得到 lepe 张量。为了避免梯度警告，使用 contiguous() 方法
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win) # 将处理后的 lepe 张量重新排列，将其从 (n, c, jh, iw) 的形状转换为 (n, jh, iw, c) 的形状，以便后续的操作

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors 通过路由器模块 router 对查询 q_win 和键 k_win 进行路由操作，得到注意力权重和索引，两者都是形状为 (n, p^2, topk) 的张量

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v) 根据路由得到的索引和权重，通过键值对选择模块 kv_gather 从键值对张量 kv_pix 中选择特定的键值对，得到形状为 (n, p^2, topk, h_kv*w_kv, c_qk+c_v) 的张量
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1) # 将选择的键值对张量按照查询键和值的维度进行分割，得到键张量 k_pix_sel 和值张量 v_pix_sel，分别为形状 (n, p^2, topk, h_kvw_kv, c_qk) 和 (n, p^2, topk, h_kvw_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here? 将键张量 k_pix_sel 重新排列为形状 (np^2, m, c, kw2)，其中 m 表示注意力头的数量，将其展平为 BMLC 的形式，以便后续的矩阵乘法操作
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m) 将值张量 v_pix_sel 重新排列为形状 (np^2, m, kw2, c)，同样展平为 BMLC 的形式
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m) 将查询张量 q_pix 重新排列为形状 (n*p^2, m, w2, c)，同样展平为 BMLC 的形式

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv) 计算注意力权重，首先对查询张量乘以缩放因子，然后与键张量进行矩阵乘法操作，得到注意力权重张量，形状为 (np^2, m, w2, kh_kv*w_kv)
        attn_weight = self.attn_act(attn_weight) # 通过 Softmax 激活函数对注意力权重进行归一化，确保注意力权重的和为1
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c) 将归一化后的注意力权重与值张量进行矩阵乘法操作，得到输出张量，形状为 (n*p^2, m, w2, c_v)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win) # 将输出张量重新排列为形状 (n, jh, iw, m*c)，其中 j 和 i 分别表示窗口的高度和宽度，h 和 w 分别表示输出的高度和宽度
        out = out + lepe # 将输出张量与侧边深度可分离卷积处理后的张量相加，用于引入局部位置信息
        # output linear
        out = self.wo(out)
        return out

class BiFormerBlock(nn.Module):
    def __init__(self, dim,
                 outdim,
                 n_win,
                 drop_path=0.,
                 layer_scale_init_value=-1,
                 num_heads=8,
                 qk_dim=None,
                 qk_scale=None,
                 kv_per_win=4,
                 kv_downsample_ratio=4,
                 kv_downsample_kernel=None,
                 kv_downsample_mode='ada_avgpool',
                 topk=4,
                 param_attention="qkvo",
                 param_routing=False,
                 diff_routing=False,
                 soft_routing=False,
                 mlp_ratio=4,
                 mlp_dwconv=False,
                 side_dwconv=5,
                 before_attn_dwconv=3,
                 pre_norm=True,
                 auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim # 如果 qk_dim 的值为 None，则将其设为 dim。这行代码用于设置查询和键的维度，如果没有显式提供 qk_dim 的值，就使用输入特征的维度 dim

        # modules
        if before_attn_dwconv > 0: # 大于 0，则创建一个卷积操作 nn.Conv2d 作为位置嵌入 pos_embed；否则，创建一个 lambda 函数，该函数接受一个参数 x 并返回 0。
            self.pos_embed = nn.Conv2d(dim, dim,  kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # important to avoid attention collapsing 用于对输入数据进行规范化处理，以避免注意力折叠（attention collapsing）
        if topk > 0: # 根据 topk 的值进行条件判断，如果大于 0，则创建一个BiLevelRoutingAttention 注意力模块 attn
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win, kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel, kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing, side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == 0: # 如果等于 0，则创建一个序列模块 nn.Sequential，其中包含一系列卷积操作，用于模拟注意力机制
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'), # compatiability
                                      nn.Conv2d(dim, dim, 1), # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim), # pseudo attention
                                      nn.Conv2d(dim, dim, 1), # pseudo out linear
                                      Rearrange('n c h w -> n h w c')# 🥭
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # 创建第二个 Layer Normalization 操作 nn.LayerNorm，用于对输出数据进行规范化处理
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio*dim)),
                                 DWConv(int(mlp_ratio*dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio*dim), dim)
                                 ) #mg # 创建一个包含线性层、深度可分离卷积、GELU 激活函数和另一个线性层的序列模块 mlp，用于实现多层感知机（MLP）部分
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # 则创建一个 DropPath 操作 DropPath,否则，创建一个恒等映射 nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True # 设置一个布尔值，表示是否使用层标准化
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) # 创建一个可学习的参数 gamma1，其值为 layer_scale_init_value 乘以一个维度为 dim 的全为 1 的张量，该参数将用于层标准化
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) # 与 gamma1 类似，用于另一种层标准化
        else:
            self.use_layer_scale = False # 如果 layer_scale_init_value 不大于 0，将 use_layer_scale 设置为 False，表示不使用层标准化
        self.pre_norm = pre_norm # 设置一个布尔值，表示是否进行预标准化
        self.outdim = outdim # 将输出维度保存在 outdim 中，这个值在之后可能会被用到


    def forward(self, x):
        # conv pos embedding
        x = x + self.pos_embed(x) # 将输入 x 与位置嵌入 pos_embed 相加，用于引入位置信息
        # permute to NHWC tensor for attention & mlp #mg
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)对张量 x 进行维度置换，将其从 (N, C, H, W) 的格式转换为 (N, H, W, C) 的格式，以便后续的注意力和多层感知机操作

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale: # 如果进行预标准化并且使用层标准化,则对输入进行规范化后，分别应用注意力和多层感知机操作，并加上 DropPath 操作
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x))) # (N, H, W, C)
            else: # 直接应用注意力和多层感知机操作，并加上 DropPath 操作
                x = x + self.drop_path(self.attn(self.norm1(x))) # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x))) # (N, H, W, C)
        else:
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x))) # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x))) # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x))) # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return x