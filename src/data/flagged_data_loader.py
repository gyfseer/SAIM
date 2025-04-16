#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : flagged_data_loader.py
    @Time   : 2024/06/21 12:23:39
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''

from ..global_var import GlobalVar
from torch.utils.data import DataLoader

class FlaggedDataLoader(DataLoader):
    """
        为dataloader包装一个控制信号, 当信号条件成立时, 退出遍历, 所有的dataloader均需要此包装类
    
    Attributes:
        dataloader: 自定义的dataloader
        flag: 标志位, 由程序控制
    
    """
    def __init__(self, dataloader):
        assert isinstance(dataloader, DataLoader), f"{dataloader} must be a class or a subclass of torch.utils.data.DataLoader"
        self.dataloader = dataloader
        assert isinstance(dataloader, DataLoader), f"{dataloader} must be a class or a subclass of torch.utils.data.DataLoader"
        self.iterator = iter(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.dataloader.__len__()
    
    def __next__(self):
        if not GlobalVar.dataloader_control_signal.is_set():
            raise StopIteration
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            raise StopIteration