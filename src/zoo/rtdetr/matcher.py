

import torch
import torch.nn.functional as F 

from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from src.core import register


@register
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2] # 获取到num_query和bs

        # We flatten to compute the cost matrices in a batch
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1)) # 从decoder_layer中拿到的类别预测输出是logits，之后做一个sigmoid再将前两个维度进行flattern展开2 * 300=600，得到的就是(600,80)
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4] # 于坐标损失同样将前两个维度展平得到(600,4)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # 从targets框中提取到真实的类别和坐标
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        if self.use_focal_loss: # 将预测类别和真实类别计算损失使用 out_prob和tgt_ids
            out_prob = out_prob[:, tgt_ids] # 因为我们不知道boundingbox具体的类别，所以既计算出作为正样本的损失，又要计算他作为负样本的损失
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob)**self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class   #然后将他作为正样本的损失减去作为负样本的损失，得到的就是最终的损失
        else: # 结果是（600,6）对应的是大的尺寸，这个6是obj数量
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) #利用预测boundingbox坐标和真实坐标计算L1损失，

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))# 利用预测boundingbox坐标和真实坐标计算GIOU损失，结果都是(600,6)
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou# 然后通过加权和矩阵计算出最终的损失值，
        C = C.view(bs, num_queries, -1).cpu() # 将尺寸从(600,6)变为(2,300,6)将大的损失矩阵分开

        sizes = [len(v["boxes"]) for v in targets] # 接下来将从矩阵中提取出有效的部分，就是有颜色的部分
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
# 通过linear_sum_assignment函数得到匈牙利匹配的最优解，
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # 得到的是匹配的索引并转化为tensor的格式
