# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step): # 前向传播
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):  # 反向传播
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    value:                 shape=[bs=1, h1*w1+h2*w2=30, 多头数=2, 隐变量的维度=2]
    value_spatial_shapes:  shape=[2,2] ---> value=[[6,4],[3,2]] 第一张特征图的h=6,w=4. 第二张特征图的h=3,w=2
    sampling_locations:    shape=[bs=1,seq_len=2,n_heads=2,特征图数=2,参考点数=2,坐标点xy=2]
    attention_weights:     shape=[bs=1,seq_len=2,n_heads=2,特征图数=2,参考点数=2]
    """
    # for debug and test only,
    # need to use cuda version instead
    batch_size, seq_len, n_heads, d_k = value.shape
    _, Lq_, n_heads, level, sp_num, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1) # [value1=[bs=1, h1*w1=24, 多头数=2, 隐变量的维度=2],  --->第1张特征图
    sampling_grids = 2 * sampling_locations - 1                                   #  value1=[bs=1, h2*w2=6, 多头数=2, 隐变量的维度=2]    --->第2张特征图
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # batch_size, H_*W_, n_heads, d_k -> batch_size, H_*W_, n_heads*d_k -> batch_size, n_heads*d_k, H_*W_ -> [batch_size*n_heads, d_k, H_, W_] #### [bs=1, hn*wn, 多头数=2, 隐变量的维度=2]-->[bs=1, hn*wn, 多头数*隐变量的维度=4]-->[bs=1, 多头数*隐变量的维度=4,hn*wn]=>[bs*多头数=2,  隐变量的维度=2,  hn=6,    wn=4]
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(batch_size*n_heads, d_k, H_, W_) 
        # batch_size, Lq_, n_heads, sp_num, 2 -> batch_size, n_heads, Lq_, sp_num, 2 -> [batch_size*n_heads, Lq_, sp_num, 2]   #### [bs=1,seq_len=2,n_heads=2,参考点数=2,坐标点xy=2]-->[bs=1, n_heads=2, seq_len=2, 参考点数=2, 坐标点xy=2]--------------------------->[bs*n_heads=2, seq_len=2, 参考点数=2, 坐标点xy=2]
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1) # 
        # batch_size*n_heads, d_k, Lq_, sp_num
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,    # --> sampling_value_l_.shape=[bs*n_heads=2, d_k=2, Lq_=2, sp_num=2]
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, Lq_, n_heads, level, sp_num) -> (batch_size, n_heads, Lq_, level, sp_num) -> (batch_size*n_heads, 1, Lq_, level*sp_num)
    attention_weights = attention_weights.transpose(1, 2).reshape(batch_size*n_heads, 1, Lq_, level*sp_num)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(batch_size, n_heads*d_k, Lq_)
    return output.transpose(1, 2).contiguous()  # shape=[batch_size, Lq_, n_heads*d_k]
