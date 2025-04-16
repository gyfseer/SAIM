#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : ious.py
    @Time   : 2024/06/22 14:44:32
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    
    iou计算相关程序
'''
import torch

# def box_ciou(box1, box2, eps: float = 1e-7):
#     """
#     Return complete intersection-over-union (Jaccard index) between two sets of boxes.
#     Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
#     ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
#     Args:
#         boxes1 (Tensor[N, 4]): first set of boxes
#         boxes2 (Tensor[M, 4]): second set of boxes
#         eps (float, optional): small number to prevent division by zero. Default: 1e-7
#     Returns:
#         Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
#         for every element in boxes1 and boxes2
    
#     参考: YOLOv5
#     """

#     def box_area(box):
#         # box = 4xn
#         return (box[2] - box[0]) * (box[3] - box[1])

#     area1 = box_area(box1.T)
#     area2 = box_area(box2.T)
    
#     inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
#     union = (area1[:, None] + area2 - inter)+ eps

#     iou = inter / union

#     return iou

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    
    Args:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)