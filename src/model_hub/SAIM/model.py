# -*- encoding: utf-8 -*-
'''
    @文件名称   : model.py
    @创建时间   : 2023/12/28 11:12:09
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 构建检测模型
                                训练: 外部提供训练图片和标注数据, 对外返回损失值
                                测试: 外部提供测试图片, 对外返回预测结果 边界框列表-标签列表-置信度列表

    @参考地址   : 无
'''

import torch
import torch.nn as nn

from .backbone import Backbone
from .adapter import BackboneNeckAdapter
from .neck import Neck 
from .head import Head
from .loss import ProgressiveLoss

import matplotlib.pyplot as plt
def draw(tensor, index, vmin=None, vmax=None):
    heatmap = tensor.cpu().numpy()
    fig, ax = plt.subplots()
    if vmin is None:
        im = ax.imshow(heatmap, cmap="hot")
    else:
        im = ax.imshow(heatmap, cmap="hot", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.savefig(f"confidence-{index}.jpg")
    plt.close()


class Model(nn.Module):

    def __init__(self, device):
        super(Model, self).__init__()
        self.backbone = Backbone()
        self.backbone_neck_adapter = BackboneNeckAdapter()
        self.neck = Neck()
        self.head = Head(16, device)
        self.loss = ProgressiveLoss(16, device)
        self.sigmoid = nn.Sigmoid()
        self.loss_value = 0
    
    def forward(self, x, instance=None, state=None,  train=True, init_state=False):
        x = self.backbone(x)
        x = self.backbone_neck_adapter(x)
        x = self.neck(x)
        x = self.head(x)
        if train:
            return  self.train(x, instance, state) # 损失值
        else:
            if init_state:
                return x
            else:
                return self.predict(x, instance, state) # 边界框列表-标签列表-置信度列表 
    
    def train(self, x, instance, state):
        return self.loss(x, instance, state)

    def predict(self, x, instance, state, thr=0.5):
        """
            Args:
                @param x: 模型预测结果
                @param instance: 图片信息,包括图片缩放信息和尺寸信息等
                @param state: 模型维护的类别状态
        """
        # draw(x[0,0,:,:], 0)
        x[:, :1, :, :] = self.sigmoid(x[:, :1, :, :])
        # draw(x[0,0,:,:], 1)
        out = x.squeeze(0)
        out = out[:, out[0, :, :]>thr].transpose(0,1)
        if out.shape[1] == 0:
            return [], [], []
        cls_values, cls_index = torch.max(out[:, 5:] @ state.get_state_matrix().transpose(0,1), dim=1)
        out[:, 0] = out[:, 0] *cls_values
        out[:, 6] = cls_index
        results = self.nms(out,  iou_thr=0.2)
        pred_bbox = []
        score = []
        label = []
        for p in results:
            if p[0,0] < thr:
                continue
            bbox = [   
                int((p[0, 1] - instance['padding'][0])/instance['ratio']),
                int((p[0, 2] - instance['padding'][1])/instance['ratio']),
                int((p[0, 3] - instance['padding'][0])/instance['ratio']),
                int((p[0, 4] - instance['padding'][1])/instance['ratio'])
            ]
            bbox[0] = bbox[0] if bbox[0] > 0 else 0
            bbox[1] = bbox[1] if bbox[1] > 0 else 0
            bbox[2] = bbox[2] if bbox[2] < instance["image_width"] else instance["image_width"]
            bbox[3] = bbox[3] if bbox[3] < instance["image_height"] else instance["image_height"]
            if bbox[0] < instance["image_width"] and bbox[1] < instance["image_height"]:
                pred_bbox.append(bbox)
                score.append(float(p[0, 0]))
                label.append(state.get_state_decode()[int(p[0, 6])])
        
        return pred_bbox, label, score

    def nms(self, output,  iou_thr=0.5):
        """Non-maximum Supression

        Args:
            output: 模型输出(c, w, h).
            iou_thr: iou置信度阈值.
        
        Returnd:
            result_list: list of bbox.
        """
        def iou(bbox1, bbox2, eps=1e-10):
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
            return iou
        output = output[torch.sort(output[:,0], descending=True)[1], :]
        result_list = []
        while output.shape[0]:
            temp = output[0, :]
            output = output[1:, :]
            if (temp[4] - temp[2])*(temp[3] - temp[1]) < 4:
                continue
            # if temp[0] < 0.5:
            #     continue
            temp = torch.unsqueeze(temp, dim=0)
            ious = iou(temp[:, 1:5], output[:, 1:5])
            ious = ious.view(-1)
            # print(output[ious>=iou_thr, :].shape[0])
            if output[ious>=iou_thr, :].shape[0]<10:
                continue
            result_list.append(temp)
            output = output[ious<iou_thr, :]

        return result_list
    