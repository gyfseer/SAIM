# -*- encoding: utf-8 -*-
'''
    @文件名称   : loss.py
    @创建时间   : 2023/12/28 10:51:16
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 渐进式匹配损失
    @参考地址   : 无
'''

import math
import torch
import torch.nn as nn

class ProgressiveLoss(nn.Module):
    
    def __init__(self, step, device):
        super(ProgressiveLoss, self).__init__()
        self.device = device
        self.step = step
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction="mean")
    
    def forward(self, x, target, state, thr=0.5):
        conf_loss_list = []
        bbox_loss_list = []
        cls_loss_list = []
        for index, item in enumerate(target):
            if state.get_name_mapping()[ item["label"][0]].endswith("background"):
                conf_loss_list.append(self.bce_with_logits(x[index, 0, :, :], torch.zeros_like(x[index, 0, :, :]).to(self.device)))
                # s = torch.sigmoid(x[index, 0, :, :])
                # print(s[s>0.5].shape)
                continue
            sorted_bbox_list, sorted_label_list = ProgressiveLoss.sort_bboxes_by_area(item['bbox'], item["label"])
            mask = torch.zeros_like(x[index, 0, :, :]).to(self.device)
            # 候选可预测掩码
            mask[torch.sigmoid(x[index, 0, :, :])>thr] = 1
            # 通过人为标注计算损失
            pr_bbox = []
            gt_bbox = []
            pr_label = []
            gt_label = []
            for i, bbox in enumerate(sorted_bbox_list):
                # 如果cy=-1, cx=-1,即未找到可预测目标的点
                cy , cx = ProgressiveLoss.search_xy_in_mask(bbox, mask, self.step)
                mask[cy, cx] = 2
                pr_bbox.append(x[index, 1:5, cy, cx].view(1, 4))
                pr_label.append(x[index, 5:, cy, cx].view(1, -1))
                gt_bbox.append(bbox)
                gt_label.append(sorted_label_list[i])
            gt_bbox = torch.tensor(gt_bbox, requires_grad=True).to(self.device)
            pr_bbox = torch.cat(pr_bbox, dim=0)
            pr_label = torch.cat(pr_label, dim=0)
            bbox_loss_list.append(-torch.mean(ProgressiveLoss.iou(pr_bbox, gt_bbox, CIoU=True)))
            conf_loss_list.append(self.bce_with_logits(x[index, 0, mask==2], torch.ones_like(x[index, 0, mask==2]).to(self.device)))
            # 计算类别损失
            gt_cls = []
            for name in gt_label:
                gt_cls.append(state.get_state()[name])
            gt_cls = torch.cat(gt_cls, dim=0)
            human_labeled_inner_loss = -torch.mean(torch.sum(pr_label*gt_cls, dim=1))
            cls_loss_list.append(human_labeled_inner_loss)
            
            # 上面计算人工标签带来的损失,下面计算负样本带来的损失
            if x[index, 0, mask==0].shape[0]:
                conf_loss_list.append(self.bce_with_logits(x[index, 0, mask==0], torch.zeros_like(x[index, 0, mask==0]).to(self.device)))
                # none_target_cls = x[index, 5:, mask==0].transpose(1,0)
                # cls_loss_list.append(torch.mean(torch.max(torch.matmul(none_target_cls, state.get_state_matrix().transpose(1, 0)), dim=1)[0]))            

            # 计算候选框带来的渐进损失
            learnt_prediction = x[index, :, mask==1].transpose(1,0)
            if learnt_prediction.shape[0]:
                gt_bbox = torch.tensor(sorted_bbox_list, requires_grad=True).to(self.device)
                cious = ProgressiveLoss.box_ciou(learnt_prediction[:, 1:5], gt_bbox)
                values, indices = torch.max(cious, dim=1)
                # print(cious)
                # 过滤掉cious < 0的情况
                if learnt_prediction[values < 0, 0].shape[0]:
                    conf_loss_list.append(self.bce_with_logits(learnt_prediction[values < 0, 0], torch.zeros_like(learnt_prediction[values < 0, 0]).to(self.device)))
                # cois >= 0的情况，采用匹配损失 
                if learnt_prediction[values >= 0, 0].shape[0]:
                    indices = indices[values >= 0]
                    pred_cls = learnt_prediction[values >= 0, 5:]
                    learnt_prediction = learnt_prediction[values >= 0, :]
                    values = values[values >= 0]
                    similarity = torch.matmul(pred_cls, state.get_state_matrix().transpose(1, 0))
                    cls_values, cls_indices = torch.max(similarity, dim=1)
                    is_match = []
                    for i in range(indices.shape[0]):
                        if sorted_label_list[int(indices[i])] == state.get_state_decode()[int(cls_indices[i])]:
                            is_match.append(True)
                        else:
                            is_match.append(False)
                    is_match = torch.tensor(is_match).to(self.device)
                    # 匹配损失
                    if cls_values[is_match].shape[0]:
                        # torch.full_like(learnt_prediction[is_match, 0], 0.9)
                        conf_loss_list.append(self.bce_with_logits(learnt_prediction[is_match, 0], torch.full_like(learnt_prediction[is_match, 0], 0.9).to(self.device)))
                        # conf_loss_list.append(self.bce_with_logits(learnt_prediction[is_match, 0], torch.ones_like(learnt_prediction[is_match, 0]).to(self.device)))
                        bbox_loss_list.append(-torch.mean(values[is_match]))
                        cls_loss_list.append(-torch.mean(cls_values[is_match]))

                    # 不匹配损失
                    # if cls_values[is_match == False].shape[0]:
                    #     conf_loss_list.append(self.bce_with_logits(learnt_prediction[is_match == False, 0], torch.zeros_like(learnt_prediction[is_match == False, 0]).to(self.device)))

        loss = 0 
        if len(cls_loss_list):
            loss += sum(cls_loss_list)
        if len(bbox_loss_list):
            loss += sum(bbox_loss_list)
        if len(conf_loss_list):
            loss += sum(conf_loss_list)

        return loss

    @staticmethod
    def sort_bboxes_by_area(bboxes, labels):
        # 计算每个矩形的面积
        areas = [abs((x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in bboxes]

        # 将(面积,索引)元组列表排序,获得索引列表 
        sorted_idx = [i[0] for i in sorted(enumerate(areas), key=lambda x:x[1])]

        # 使用索引列表对bboxes和labels同时排序
        sorted_bboxes = [bboxes[i] for i in sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        return sorted_bboxes, sorted_labels

    @staticmethod
    def search_xy_in_mask(bbox, mask, step):
        l, t, r, b = bbox
        cx = round((l+r)/2/step)
        cy = round((t+b)/2/step)
        h, w = mask.shape
        cy = cy if cy < h else cy-1
        cx = cx if cx < w else cx-1
        if mask[cy, cx] != 2:
            return cy, cx
        # 矩形框中心重叠
        max_edge = max(r-l, b-t)
        delta_x = (r-l)/max_edge
        delta_y = (b-t)/max_edge
        s = 1
        while True :
            # 沿左上搜索
            x = round(cx-s*delta_x)
            y = round(cy-s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break 
            # 沿右上搜索
            x = round(cx+s*delta_x)
            y = round(cy-s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            # 沿左下搜索
            x = round(cx-s*delta_x)
            y = round(cy+s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            # 沿右下搜索
            x = round(cx+s*delta_x)
            y = round(cy+s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            s += 1
            if s == 100:
                cy = -1
                cx = -1
                break
        return cy, cx



    @staticmethod
    def iou(bbox1, bbox2, GIoU=False, DIoU=False, CIoU=False, eps=1e-10):
            """Compute IoU loss of bbox1(1, 4) and bbox2(n, 4) -> (n, 1) or
            Compute IoU loss of bbox1(n, 4) and bbox2(n, 4) -> (n, 1) 

            Args:
                bbox1: a tensor with shape(1, 4) or shape(n, 4).
                bbox2: a tensor with shape(n, 4).
                GIoU,DIoU,CIoU: switch to choose loss type.
            
            Returns:
                iou: iou value list with shape(n, 1).
            
            参考: YOLOv5
            
            """
            l1, t1, r1, b1 = bbox1.chunk(4, 1)
            l2, t2, r2, b2 = bbox2.chunk(4, 1)
            w1, h1 = r1 - l1, b1 - t1
            w2, h2 = r2 - l2, b2 - t2

            # Intersection area
            inter = (torch.min(r1, r2) - torch.max(l1, l2)).clamp(0) * (torch.min(b1, b2) - torch.max(t1, t2)).clamp(0)

            # Union area
            union = w1 * h1 + w2 * h2 - inter + eps

            # IoU
            iou = inter / union
            if GIoU or DIoU or CIoU:
                cw = torch.max(r1, r2) - torch.min(l1, l2)
                ch = torch.max(b1, b2) - torch.min(t1, t2)
                if DIoU or CIoU:
                    c2 = cw**2 + ch**2 + eps
                    rho2 = ((l2 + r2 - l1 - r1) ** 2 + (b2 + t2 - b1 - t1) ** 2) / 4
                    if CIoU:
                        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                        with torch.no_grad():
                            alpha = v / (v - iou + (1 + eps))
                        return iou - (rho2 / c2 + v * alpha)  # CIoU
                    return iou-rho2/c2 # DIoU
                c_area = cw * ch + eps
                return iou - (c_area - union)/c_area # GIoU
            return iou # IoU    
    
    @staticmethod
    def box_ciou(box1, box2, eps: float = 1e-7):
        """
        Return complete intersection-over-union (Jaccard index) between two sets of boxes.
        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        Args:
            boxes1 (Tensor[N, 4]): first set of boxes
            boxes2 (Tensor[M, 4]): second set of boxes
            eps (float, optional): small number to prevent division by zero. Default: 1e-7
        Returns:
            Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
            for every element in boxes1 and boxes2
        
        参考: YOLOv5
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        union = (area1[:, None] + area2 - inter)+ eps

        iou = inter / union

        lti = torch.min(box1[:, None, :2], box2[:, :2])
        rbi = torch.max(box1[:, None, 2:], box2[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps

        # centers of boxes
        x_p = (box1[:, None, 0] + box1[:, None, 2]) / 2
        y_p = (box1[:, None, 1] + box1[:, None, 3]) / 2
        x_g = (box2[:, 0] + box2[:, 2]) / 2
        y_g = (box2[:, 1] + box2[:, 3]) / 2
        # The distance between boxes' centers squared.
        centers_distance_squared = (x_p - x_g) ** 2 + (y_p - y_g) ** 2

        w_pred = box1[:, None, 2] - box1[:, None, 0]
        h_pred = box1[:, None, 3] - box1[:, None, 1]

        w_gt = box2[:, 2] - box2[:, 0]
        h_gt = box2[:, 3] - box2[:, 1]

        v = (4 / (torch.pi ** 2)) * torch.pow((torch.atan(w_gt / (h_gt+eps)) - torch.atan(w_pred / (h_pred+eps))), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        return iou - (centers_distance_squared / diagonal_distance_squared) - alpha * v
