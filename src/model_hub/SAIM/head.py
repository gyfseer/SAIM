# -*- encoding: utf-8 -*-
'''
    @文件名称   : head.py
    @创建时间   : 2023/12/27 17:29:04
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   :  回归融合后的特征，给出模型预测输出
    @参考地址   : 无
'''

import torch
import torch.nn as nn

from .components import Conv

class Head(nn.Module):

    def __init__(self, step, device):
        super(Head, self).__init__()
        self.step = step
        self.device = device
        # 边界框回归模块
        self.bbox_modules = nn.ModuleList()
        self.bbox_modules.append(Conv(256, 256))
        self.bbox_modules.append(Conv(256, 256))
        self.bbox_modules.append(Conv(256, 128))
        self.bbox_modules.append(Conv(128, 64))
        self.bbox_modules.append(nn.Conv2d(64, 4, 1))
        # 类别特征回归
        self.cls_modules = nn.ModuleList()
        self.cls_modules.append(Conv(256, 256))
        self.cls_modules.append(Conv(256, 256))
        self.cls_modules.append(Conv(256, 128))
        self.cls_modules.append(Conv(128, 64))
        self.cls_modules.append(nn.Conv2d(64, 64, 1))
        # 置信度回归 
        self.conf_modules = nn.ModuleList()
        self.conf_modules.append(Conv(256, 256))
        self.conf_modules.append(Conv(256, 256))
        self.conf_modules.append(Conv(256, 128))
        self.conf_modules.append(Conv(128, 64))
        self.conf_modules.append(nn.Conv2d(64, 1, 1))

        self.elu = nn.ELU()
    
    def norm(self, x):
        """
            单位化张量
        """
        return x/torch.sqrt(torch.sum(x*x,dim=1).unsqueeze(1))
    
    def forward(self, x):
        out_bbox = x
        out_cls = x
        out_conf = x
        # 回归边界框
        for module_box in self.bbox_modules:
            out_bbox = module_box(out_bbox)
        
        # 回归类别特征表示
        for module_cls in self.cls_modules:
            out_cls = module_cls(out_cls)
        # 单位化
        out_cls = self.norm(out_cls)

        # 回归
        for module_conf in self.conf_modules:
            out_conf = module_conf(out_conf)

        # map coordinates from receptive field to image
        x_offset = torch.linspace(0, out_bbox.shape[-1]-1, steps=out_bbox.shape[-1])
        y_offset = torch.linspace(0, out_bbox.shape[-2]-1, steps=out_bbox.shape[-2])
        x_grid, y_grid = torch.meshgrid(x_offset, y_offset, indexing="xy")
        x_grid = x_grid.to(self.device)
        y_grid = y_grid.to(self.device)

        # 坐标转换成 (l,t,r,b)
        index = 0
        out_bbox_l = x_grid*self.step + self.step//2 + out_bbox[:, index:index+1, :, :]
        out_bbox_t = y_grid*self.step + self.step//2 + out_bbox[:, index+1:index+2, :, :]
        out_bbox_r = out_bbox_l + 1.0 + self.elu(out_bbox[:, index+2:index+3, :, :])
        out_bbox_b = out_bbox_t + 1.0 + self.elu(out_bbox[:, index+3:index+4, :, :])
        
        # 拼接  (c, l, t, r, b, cls)
        out = torch.cat([out_conf, out_bbox_l, out_bbox_t, out_bbox_r, out_bbox_b, out_cls], dim=1)

        return out