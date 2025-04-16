# -*- encoding: utf-8 -*-
'''
    @文件名称   : adapter.py
    @创建时间   : 2023/12/27 17:08:33
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 将骨干网络输出的特征在特征维度上转化为统一维度
    @参考地址   : 无
'''

import torch.nn as nn

class BackboneNeckAdapter(nn.Module):

    def __init__(self, channels=[1024, 2048], output_channel=256, kernel=1):
        super(BackboneNeckAdapter,self).__init__()
        self.convert_module_list = nn.ModuleList()
        for ch in channels:
            self.convert_module_list.append(
                nn.Sequential(
                    nn.Conv2d(ch, output_channel, kernel),
                    nn.BatchNorm2d(output_channel),
                    nn.SiLU()
                )
            )

    def forward(self, x):
        
        out = []

        for i in range(len(x)):
            out.append(self.convert_module_list[i](x[i]))
            
        return out