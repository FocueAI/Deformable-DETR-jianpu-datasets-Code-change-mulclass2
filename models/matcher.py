# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module): # 里面使用匈牙利算法 做二分图匹配
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # 2 ---超参数
        self.cost_bbox = cost_bbox   # 5 ---超参数
        self.cost_giou = cost_giou   # 2 ---超参数
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

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
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid() # shape=[bs=2 * query_num=300, class_num=91]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets]) # shape=[obj的数量]
            tgt_bbox = torch.cat([v["boxes"] for v in targets]) # shape=[obj的数量,4]

            # Compute the classification cost. ------------------------------------------------------------------------------------后面需要认真的调试以下代码----------------- 2023-5-31
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())  # 这里是 focal loss ------> shape=[600, 91]
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())      # 这里是 focal loss ------> shape=[600, 91]
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]   # ----------------------------------------------------------------> cost_class.shape=[600,6]
                           #  shape=[600,6]                shape=[600,6]
            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # out_bbox.shape=[batch_size * num_queries=600, 4], tgt_bbox.shape=[6,4] ---------------> cost_bbox.shape=[600,6]

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), # out_bbox.shape=[batch_size * num_queries=600, 4]  ----------------------> cost_giou.shape=[600,6]
                                             box_cxcywh_to_xyxy(tgt_bbox)) # tgt_bbox.shape=[该batch中box的个数=6,4]

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # C.shape=[600,6]
            C = C.view(bs, num_queries, -1).cpu() # C.shape=[bs=2,num_queries=300,6]

            sizes = [len(v["boxes"]) for v in targets] # 由于batch_size=2, sizes=[第1张图像中的目标数=2,第2张图像中的目标数=4]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] # [c.shape for i, c in enumerate(C.split(sizes, -1))].shape=[shape1=[2,300,2],shape2=[2,300,4]] --->实现二分图匹配
            # indices = [(array([ 70, 254]), array([0, 1])), (array([ 79,  89, 150, 275]), array([2, 3, 0, 1]))]
            #  在第1张图中: 生成的第70号box 匹配 第0号真实box. 生成的第254号box 匹配 第1号真实box
            #  在第2张图中: 生成的第79号box 匹配 第2号真实box. 生成的第89号box 匹配 第3号真实box. 生成的第150号box 匹配 第0号真实box. 生成的第275号box 匹配 第1号真实box
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, # 2
                            cost_bbox=args.set_cost_bbox,   # 5
                            cost_giou=args.set_cost_giou)   # 2
