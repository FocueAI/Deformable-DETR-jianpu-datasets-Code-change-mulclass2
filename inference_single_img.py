# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import argparse
import datetime
import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import torchvision.transforms as T

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util import box_ops
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import time

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='weight_result_6_9/checkpoint0044.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--img_path', type=str, help='input image file for inference')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 标签
PRE_DEFINE_CATEGORIES = { # 有json文件映射而成的
   "5":0,
   "2":1,
   "1":2,
   "3":3,
   "7":4,
   "6":5,
   "4":6
#    "7":7,
#    "x":8,
#    "yy":9
}
index2classtr = {value:key for key,value in PRE_DEFINE_CATEGORIES.items()}


index2cls_a = {
    0:'上下无点',
    1:'下面有1点',
    2:'下面有2点',
    3:'上面有1点',
    4:'上面有2点'
}

index2cls_b = {
    0:'右边无点',
    1:'右边有1点',
}

index2cls_c = {
    0:'下面无线',
    1:'下面有1条线',
    2:'下面有2条线'
}

index2cls_d = {
    0:'无上升下降符号',
    1:'有上升符号',
    2:'有下降符号'
}

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    t0 = time.time()
    im = Image.open(args.img_path)
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    img=img.cuda()
    # propagate through the model
    outputs = model(img)
    # [2, 300, 91], [2, 300, 4] 
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    ################################################ 新增加的内容 ##########################################
    out_logits_a, out_logits_b, out_logits_c, out_logits_d = outputs['pred_a_logits'], outputs['pred_b_logits'], outputs['pred_c_logits'], outputs['pred_d_logits']
    prob = out_logits.sigmoid()      # [1, 300, 91]
    prob_a = out_logits_a.sigmoid()  # [1, 300, 5]
    prob_b = out_logits_b.sigmoid()  # [1, 300, 2]
    prob_c = out_logits_c.sigmoid()  # [1, 300, 3]
    prob_d = out_logits_d.sigmoid()  # [1, 300, 3]
    # shape=[1,100], [1,100]                             (batch_size=1, 300*91=27300) 
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)   # 前 100 位的 类别最大概率对应的列表
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2] # out_logits.shape=[1,300,91]---> //91  # 前 100 位 类别概率对应box 序号列表
    labels = topk_indexes % out_logits.shape[2]      # 前 100 位 类别,对应的标签序号列表
    ########################################################
    # topk_values, topk_indexes = torch.topk(prob_a.view(out_logits_a.shape[0], -1), 100, dim=1)
    # scores_ = topk_values
    # topk_boxes = topk_indexes // out_logits_a.shape[2] # out_logits.shape=[1,300,91]---> //91
    # labels_a = topk_indexes % out_logits_a.shape[2]  
    # labels_b = topk_indexes % out_logits_b.shape[2]
    # labels_c = topk_indexes % out_logits_c.shape[2]
    # labels_d = topk_indexes % out_logits_d.shape[2]
    
    # prob_a.view(out_logits_a.shape[0], -1)[:,topk_boxes]
    label_a_list = torch.argmax(prob_a[0][topk_boxes],dim=2)
    label_b_list = torch.argmax(prob_b[0][topk_boxes],dim=2)
    label_c_list = torch.argmax(prob_c[0][topk_boxes],dim=2)
    label_d_list = torch.argmax(prob_d[0][topk_boxes],dim=2)
    
    ########################################################
    
    
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

    keep = scores[0] > 0.35
    boxes = boxes[0, keep]
    labels = labels[0, keep]
    
    labels_a = label_a_list[0, keep]
    labels_b = label_b_list[0, keep]
    labels_c = label_c_list[0, keep]
    labels_d = label_d_list[0, keep]

    # and from relative [0, 1] to absolute [0, height] coordinates
    im_h,im_w = im.size
    #print('im_h,im_w',im_h,im_w)
    target_sizes =torch.tensor([[im_w,im_h]])
    target_sizes =target_sizes.cuda()
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    print(time.time()-t0)
    #plot_results
    source_img = Image.open(args.img_path).convert("RGBA")

    draw = ImageDraw.Draw(source_img)
    
    # print("Boxes",boxes,boxes.tolist())
    ##################### 将预测的box排序,其他内容跟着他走 begin ###########################
    boxes_info = []
    for box_coord, label_index, label_a, label_b, label_c, label_d in zip(boxes[0].tolist(), labels.tolist(), labels_a.tolist(), labels_b.tolist(), labels_c.tolist(), labels_d.tolist() ):
            xmin, ymin, xmax, ymax = box_coord
            boxes_info.append([xmin, ymin, xmax, ymax, label_index, label_a, label_b, label_c, label_d])
    boxes_info = sorted(boxes_info,key=lambda x: x[0])
    ##################### 将预测的box排序,其他内容跟着他走 end ###########################
    
    print('=='*6,"以下是预测的结果",'=='*6)
    for xmin, ymin, xmax, ymax, label_index, label_a, label_b, label_c, label_d in boxes_info:
        clstr = index2classtr[label_index]
        
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline ="red")
        draw_text_coord = ((xmin + xmax)/2,ymin-2)
        draw.text(draw_text_coord,clstr,(0,255,0))
        print(f'str:{clstr}, label_a:{index2cls_a[label_a]},label_b:{index2cls_b[label_b]},label_c:{index2cls_c[label_c]},label_d:{index2cls_d[label_d]}')

    source_img.save('test.png', "png")
    print('--'*6,"预测完毕",'--'*6)
    # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    # print("Outputs",results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)