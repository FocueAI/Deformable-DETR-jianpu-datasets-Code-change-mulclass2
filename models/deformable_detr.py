# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        ######################### 新增的其他分类-begin ##########################
        self.class_a = nn.Linear(hidden_dim, 5) # 对应 简谱 音符上下的点的个数
        self.class_b = nn.Linear(hidden_dim, 2) # 对应 简谱 音符右边点的个数
        self.class_c = nn.Linear(hidden_dim, 3) # 对应 简谱 音符下边的线数
        self.class_d = nn.Linear(hidden_dim, 3) # 对应 简谱 音符左边的升降号的
        
        ######################### 新增的其他分类-end ############################
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3) # TODO: 搞清楚这里面的含义
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1: # num_feature_levels=4 ,执行该分支
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs): # num_backbone_outs=3
                in_channels = backbone.num_channels[_] # backbone.num_channels = [512,1024,2048]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else: # 平时应该是执行该分支
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            
            ######################### 新增的其他分类-begin ########################
            self.class_a_embed = nn.ModuleList([self.class_a for _ in range(num_pred)])
            self.class_b_embed = nn.ModuleList([self.class_b for _ in range(num_pred)])
            self.class_c_embed = nn.ModuleList([self.class_c for _ in range(num_pred)])
            self.class_d_embed = nn.ModuleList([self.class_d for _ in range(num_pred)])
            ######################### 新增的其他分类-end ##########################
            
            
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None   

        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):               # 不执行该分支
            samples = nested_tensor_from_tensor_list(samples)
        # ------------------------------------- step1: CNN特征提取
        features, pos = self.backbone(samples) # samples.tensors.shape=[2,3,608,822], samples.mask.shape=[2,608,822]
        # features = [(mask.shape=[2,76,103],tensors.shape=[2,512,76,103]), (mask.shape=[2,38,52],tensors.shape=[2,1024,38,52]), (mask.shape=[2,19,26],tensors.shape=[2,2048,19,26])] # 其中的mask是使用F.interpolate做下采样处理
        # pos = [[2, 256, 76, 103], [2, 256, 38, 52], [2, 256, 19, 26] ]
        srcs, masks = [], []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):  # 4 > 3 --> True # 当特征金字塔级别数(设置的超参数) > 特征图的数量(renet最后输出的特征图数量)  ----->
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors) # 就用最后一个特征图 ----> 1*1卷积 投影成一个新的特征图
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        # ------------------------------------- step2: deformable-Transformer 登上舞台
        query_embeds = None
        if not self.two_stage: # 执行该分支
            query_embeds = self.query_embed.weight  # nn.Embedding(num_queries, hidden_dim*2)
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs, # srcs.shape  = [[2,256,76,103],[2,256,38,52],[2,256,19,26],[2,256,10,13]]
                                                                                                            masks,# masks.shape = [[2,76,103],    [2,38,52],    [2,19,26],    [2,10,13]]
                                                                                                            pos,  # pos.shape   = [[2,256,76,103],[2,256,38,52],[2,256,19,26],[2,256,10,13]]
                                                                                                            query_embeds # quer_embeds = 有300个元素,每个元素的shape=(512,)
                                                                                                            )   # 可形变的注意力魔魁啊
        # hs.shape=[decode_num=6,bs=2,query_num=300,d_model=256], init_reference.shape=[bs=2,query_num=300,xy对应的坐标=2], inter_references.shape=[6,2,300,2], enc_outputs_class=None, enc_outputs_coord_unact=None
        outputs_classes = []
        outputs_a_classes, outputs_b_classes, outputs_c_classes, outputs_d_classes = [], [], [], []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference                 # shape=[2, 300, 2]
            else:
                reference = inter_references[lvl - 1]      # shape=[2, 300, 2]
            reference = inverse_sigmoid(reference)         # shape=[2, 300, 2]
            outputs_class = self.class_embed[lvl](hs[lvl]) # self.class_embed[lvl] = nn.Linear(hidden_dim=256, num_classes=91)  #### hs[lvl].shape=[2,300,256]======>outputs_class.shape=[2,300,class_num=91]
            
            ################## 添加其他分类分支-begin ###############
            outputs_a_class = self.class_a_embed[lvl](hs[lvl]) # shape=[2, 300, 5]
            outputs_b_class = self.class_b_embed[lvl](hs[lvl]) # shape=[2, 300, 2]
            outputs_c_class = self.class_c_embed[lvl](hs[lvl]) # shape=[2, 300, 3]
            outputs_d_class = self.class_d_embed[lvl](hs[lvl]) # shape=[2, 300, 3]
            ################## 添加其他分类分支-end #################
            
            tmp = self.bbox_embed[lvl](hs[lvl]) # self.bbox_embed[lvl]---->[Linear(256,256),Linear(256,256),Linear(256,4)] # hs[lvl].shape=[2,300,256] ======> tmp.shape=[2,300,4] 我估计4的含义是(中心点x的偏移量,中心点y的偏移量,box的宽,box的高)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference   # 相当于参考点+中心点的偏移量
            outputs_coord = tmp.sigmoid()               # outputs_coord.shape=[2,300,4]
            outputs_classes.append(outputs_class)       
            outputs_coords.append(outputs_coord)
            
            ################## 添加其他分类分支-begin ###############
            outputs_a_classes.append(outputs_a_class)
            outputs_b_classes.append(outputs_b_class)
            outputs_c_classes.append(outputs_c_class)
            outputs_d_classes.append(outputs_d_class)
            ################## 添加其他分类分支-end #################
            
            
        outputs_class = torch.stack(outputs_classes)    # outputs_class.shape=[6,2,300,91]
        outputs_coord = torch.stack(outputs_coords)     # outputs_class.shape=[6,2,300, 4]
        
        ################## 添加其他分类分支-begin ###############
        outputs_a_class = torch.stack(outputs_a_classes)  # outputs_class.shape=[6,2,300,5]
        outputs_b_class = torch.stack(outputs_b_classes)  # outputs_class.shape=[6,2,300,2]
        outputs_c_class = torch.stack(outputs_c_classes)  # outputs_class.shape=[6,2,300,3]
        outputs_d_class = torch.stack(outputs_d_classes)  # outputs_class.shape=[6,2,300,3]
        ################## 添加其他分类分支-end #################

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_a_logits':outputs_a_class[-1], 'pred_b_logits':outputs_b_class[-1],'pred_c_logits':outputs_c_class[-1],'pred_d_logits':outputs_d_class[-1] } # outputs_class[-1].shape=[2,300,91], outputs_coord[-1]=[2,300,4] ========> 使用的是最后一层的数据
        if self.aux_loss: # 执行该分支
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_a_class, outputs_b_class, outputs_c_class, outputs_d_class) # ========================================================================> 使用的是前5层(一共6层,也就是除了最后一层的的)的数据

        if self.two_stage: # false
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out # {'pred_logits':shape=[2,300,91],'pred_boxes':shape=[2,300,4], 'aux_outputs':[{'pred_logits':shape=[2,300,91],'pred_boxes':shape=[2,300,4]},..这样的字典重复5次]}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_a_class, outputs_b_class, outputs_c_class, outputs_d_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_a_logits': c, 'pred_b_logits': d, 'pred_c_logits': e, 'pred_d_logits': f}                      # outputs_class.shape=[6,2,300,91],       outputs_coord.shape=[6,2,300,4]
                for a, b, c, d, e, f in zip(outputs_class[:-1], outputs_coord[:-1], outputs_a_class[:-1], outputs_b_class[:-1], outputs_c_class[:-1], outputs_d_class[:-1])] # outputs_class[:-1].shape=[5,2,300,91],  outputs_coord[:-1].shape=[5,2,300,4]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
    
    def other_cls_labels(self, outputs, targets, indices, num_boxes, log=True):
        """ 计算音符旁边的修饰符号的类别       
        """
        other_cls_a_num, other_cls_b_num, other_cls_c_num, other_cls_d_num = 5, 2, 3, 3
        losses = dict()
        # ----------------------------------------------- 类别 a ------------------------------
        src_logits = outputs['pred_a_logits']  # 修改1   -----> shape=[2, 300, 5]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels_a"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], other_cls_a_num, # 修改2   ------> shape=[2,300]
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce_a': loss_ce} # 修改3
        losses['loss_ce_a'] = loss_ce
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_a'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]  # 修改4
            
        # ----------------------------------------------- 类别 b ------------------------------
        src_logits = outputs['pred_b_logits'] # 修改1

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels_b"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], other_cls_b_num, # 修改2
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce_b': loss_ce} # 修改3
        losses['loss_ce_b'] = loss_ce
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_b'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]  # 修改4   
            
        # ----------------------------------------------- 类别 c ------------------------------
        src_logits = outputs['pred_c_logits'] # 修改1

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels_c"][J] for t, (_, J) in zip(targets, indices)]) 
        target_classes = torch.full(src_logits.shape[:2], other_cls_c_num, # 修改2
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce_c': loss_ce} # 修改3
        losses['loss_ce_c'] = loss_ce
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_c'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # 修改4               
            
        # ----------------------------------------------- 类别 d ------------------------------
        src_logits = outputs['pred_d_logits'] # 修改1

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels_d"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], other_cls_d_num, # 修改2
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce_d': loss_ce} # 修改3
        losses['loss_ce_d'] = loss_ce
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_d'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # 修改4               
            
        return losses
    
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'other_cls_labels': self.other_cls_labels,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}  # 只要 decoder最后一层的输出

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # outputs_without_aux={'pred_logits':shape=[2,300,91], 'pred_boxes':shape=[2,300,4]}
        # indices = [(array([ 70, 254]), array([0, 1])), (array([ 79,  89, 150, 275]), array([2, 3, 0, 1]))] ,具体解释详解 函数内部实现                                                
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)  # 6, 该批次中的2张图一共拥有的目标是 6个
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device) # num_boxes=tensor([6.])
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs: # 执行该分支,使用辅助损失计算(也就是说在decoder的中间层也做损失计算)
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs: # 不经过该分支
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic": # 这个是分割数据集 
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args) # CNN

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef, 'loss_ce_a': args.cls_loss_coef, 'loss_ce_b': args.cls_loss_coef, 'loss_ce_c': args.cls_loss_coef, 'loss_ce_d': args.cls_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss: # 程序走该分支
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'other_cls_labels']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha) # 该类继承了 nn.Module
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()} # 该类继承了 nn.Module
    if args.masks: # 不执行该分支
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
