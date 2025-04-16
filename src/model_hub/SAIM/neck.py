# -*- encoding: utf-8 -*-
'''
    @文件名称   : neck.py
    @创建时间   : 2023/12/27 17:14:12
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 融合不同尺度下的特征, 采用双线性插值,通过可学习的加权策略融合
    @参考地址   : 无
'''
import torch
import torch.nn as nn

from .components import Conv

class Neck(nn.Module):

    def __init__(self, step=[2]):
        """

        Args:
            step: 插值倍数列表.
        
        """
        super(Neck, self).__init__()
        self.weight = nn.Sequential(
                Conv(256, 256),
                Conv(256, 256),
                Conv(256, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        self.upsample = [
            nn.Upsample(scale_factor=s, mode="bilinear") for s in step
        ]

    def forward(self, x):
        """
        Args:
            x: 维度相同，尺寸不同的特征列表
        
        Returns:
            out: 融合后的特征

        """

        # 上菜样，双线性插值
        for i in range(1, len(self.upsample)+1):
            x[i] = self.upsample[i-1](x[i])
        # 计算特征加权权重
        weighted = [ None for i in x]
        for i in range(len(x)):
            weighted[i] = self.weight(x[i])
        weighted = self.softmax(torch.cat(weighted, dim=1))

        # 加权融合
        out = sum([x[i]*weighted[ :, i:i+1, :, :] for i in range(len(x))])

        return out
        
