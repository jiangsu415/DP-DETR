

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh



def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_queries,
                                             class_embed,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t['labels']) for t in targets] # 获取每张图像中label的数量（bs有关）里面是obj的数量
    device = targets[0]['labels'].device
    
    max_gt_num = max(num_gts) # max_gt_num之后对于这个bs中的所有图像object的尺寸都会统一为5，因为bs每张图像中最大obj数量为5，所以要把其他所有图像都填充为5  小于5的就用mask给标记上
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num # num_group为分组的个数
    num_group = 1 if num_group == 0 else num_group
    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device) # 要创建每个组，先生成一个[2,5]的矩阵初始化为类别数量
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device) # 生成一个[2,5,4]的矩阵（bs,objects,(x,y,w,h)）
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device) # pad_gt_mask也初始化为[2,5]矩阵，之后用于标记哪些位置真的有object.

    for i in range(bs): # 在通过for循环将标签和坐标信息放到初始化的两个矩阵中
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['labels']
            input_query_bbox[i, :num_gt] = targets[i]['boxes']
            pad_gt_mask[i, :num_gt] = 1 # pad_gt_mask[i, :num_gt] = 1把有物体的标为1
    # each group has positive and negative queries. 之后会复制40份，每个组内会有正负5个所以复制两份，然后一共有20个组
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device) # 接下来会生成正负标记，先生成一个[2,10,1]的矩阵，也就是成一个组的mask，然后将后面5个negative的值都变为1
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1]) # 在生成20份就是negative_gt_mask
    positive_gt_mask = 1 - negative_gt_mask # 对于positive_gt_mask会先对negative_gt_mask取反，再乘pad_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1] # dn_positive_idx提取positive_gt_mask不为0的索引,一共是有120个（1+5）*20
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts]) # 再根据每张图像正样本的数量再拆开，之后在计算denosing的数量一共是200个
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)  # 200

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5) # 进行加噪操作先从(0,1)中随机抽取[2,200]个数，如果数值小于0.5 * 0.5就把mask值置为true
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype) # 随机的采样出[2,200]个类别标签，然后将25%标签值替换为非真实标签
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class) # 替换为标签后，所有的[2,200]的标签值都变成256维的向量，得到的尺寸就是[2,200,256]也就是之后要concat的数值了

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox) # 对坐标进行一个加噪，首先对坐标做一个变换变为(x,y,x,y)存在known_bbox
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale # diff变为w/2,h/2用于对后面xmin,ymin,xmax,ymax进行计算
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0 # 之后随机生成（-1，+1）也就是后面的+或-
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask) # 随机生成一个系数（0,1）之间，先是对于后五个negative_query要乘（1,2）而前面的正标签乘（0,1）
        rand_part *= rand_sign # 再乘以随机的正负号
        known_bbox += rand_part * diff # 再让结果乘h/2或者w/2
        known_bbox.clip_(min=0.0, max=1.0) # 下面就是处理边界情况，防止出界
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox) # 然后在对x,y,x,y转化为x,y,w,h的形式
        input_query_bbox = inverse_sigmoid(input_query_bbox) # 最后再将x,y,w,h映射回原始值

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    input_query_class = class_embed(input_query_class) # [2,200,256] 将label数值编码为embedding向量

    tgt_size = num_denoising + num_queries # 先将前面定义的query的数量和加噪之后的数量做一个相加就得到了
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device) # attention mask的尺寸就是500 * 500
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True # 所有的都填充为TRUE
    
    # reconstruct cannot see each other对一个每一个group的query是否和其他的group的querymatching apart计算attention
    for i in range(num_group): # 不需要计算的地方需要打上mask值为true之后将数据打包为dn_meta返回回去
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
        
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_class, input_query_bbox, attn_mask, dn_meta #  加噪后的200queryclass,加噪后的200query_bbox,attn_mask, dn_meta
