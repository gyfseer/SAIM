#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : saim_dataloader.py
    @Time   : 2024/06/21 14:24:46
    @Authors : Jun Yang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
    @Description: write here
'''

from torch.utils.data import DataLoader

from .saim_dataset import SAIMDataset
from ...data.flagged_data_loader import FlaggedDataLoader

class SAIMDataloader:

    dataset = SAIMDataset()

    @staticmethod
    def get_dataloader():
        SAIMDataloader.dataset.prob_based_sample()
        dataloader = []
        if len(SAIMDataloader.dataset):
            dataloader = DataLoader(SAIMDataloader.dataset, batch_size=1, shuffle=True, collate_fn=lambda x:x, num_workers=8, pin_memory=True)
            dataloader = FlaggedDataLoader(dataloader)

        return dataloader