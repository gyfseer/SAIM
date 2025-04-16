# -*- encoding: utf-8-
'''
    @文件名称   : backbone.py
    @创建时间   : 2023/12/27 16:19:46
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 模型骨干网络, 从torch.hub中加载预训练好的模型作为模型初始状态 
    @参考地址   : 无
'''

import torch
import torch.nn as nn

class Backbone(nn.Module):
    """
        骨干网络模型, 用于图像特征提取.

        Atrributes:
            @param backbone_name: 骨干网络模型名称
            @param module_names: 选择输出的特征, 作为下一个模块的输入

    """
    def __init__(self, backbone_name="resnet50", module_names=["layer4", "avgpool"]):
        super(Backbone, self).__init__()

        # 加载模型
        try:
            # 从本地加载
            repo = "/home/yangjun/.cache/torch/hub/vision-0.13.0"
            self.model = torch.hub.load(repo, model=backbone_name, pretrained=True, source="local") # load backbone from torch.hub
        except FileNotFoundError:
            # 远程仓库加载
            repo = "pytorch/vision:v0.13.0"
            self.model = torch.hub.load(repo, model=backbone_name, pretrained=True) # load backbone from torch.hub

        self.output = []  # 存储骨干网络输出
        
        # 注册钩子函数,收集不同阶段的特征输出 
        for module_name, module in self.model.named_modules():
            if module_name in module_names:
                module.register_forward_hook(self.make_hook())
        

    def make_hook(self):
        """
        钩子函数,收集模型输出特征


        Returns:
            hook: 钩子函数句柄
        
        """
        def hook(module, in_module, out_module):
            self.output.append(in_module[0])

        return hook

    def forward(self, x):
        """
        模型前向过程

        Args:
            x: 批量图片.[B, N, H, W]

        
        Returns:
            self.output: 骨干网络模型输出.

        """
        self.output = []  
        self.model(x)
        
        return self.output
