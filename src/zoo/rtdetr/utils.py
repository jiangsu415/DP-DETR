"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape
# 这里还是对value进行处理
    split_shape = [h * w for h, w in value_spatial_shapes] # # 先获取每个值，计算一下每个特征图中的query数量
    value_list = value.split(split_shape, dim=1) # 将memory中的6804按照每个特征图的query数量进行拆开，得到值存储到value_list中
    sampling_grids = 2 * sampling_locations - 1 # 为了将计算出的keys的坐标重新分配一下，# 为了将预测出的sampling_locations从左边变为右边是因为使用的是F.grid_sample方法。
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute( # 之后通过for循环迭代出每个特征图尺寸，之后提取出每个特征图的value经过一系列reshape操作
            0, 2, 1).reshape(bs * n_head, c, h, w) # (2,5184,8,32)->(2,5184,256)->(2,256,5184)->(16,32,72,72)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1) # (2,500,8,3,4,2)->(2,500,8,4,2)->(2,8,500,4,2)->(16,500,4,2)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample( # 之后经过F.grid_sample提取出500个query对应的keys，得到的就是(16,32,500,4)，16是两个特征图乘8个head,32是每个value的特征向量值长度，500表示query的数量，4表示每个query都预测出4个keys。
            value_l_,                      # 取出sample_locations对应的value
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_) # 之后将这三个都append到sampling_value_list中，再进行一个reshape操作方便后续计算
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points) # reshape一下attn_weig的形状方便之后去计算
    output = (torch.stack(#  之后在sampling_value_list倒数第二个维度（三个尺度）进行一个stack操作粘在一起，再将3 * 4flattern开为12，再让attention_weights乘value就是下面这个部分
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q) # # 再对这个结果计算3个sum，也就是三个layer的3 * 4的结果，最后再reshape一下得到就是(2,500,256)的输出
        #最后的乘Wm是返回之后在output = self.output_proj(output)中做的，经过一个全连接层输出依旧是(2,500,256)，回到最最后
    return output.permute(0, 2, 1)


import math 
def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


