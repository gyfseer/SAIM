#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : saim_dataset.py
    @Time   : 2024/06/21 14:26:56
    @Authors : Jun Yang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
    @Description: write here
'''

import os

from PIL import Image

from .augment import transform
from ...data.base_dataset import BaseDataset

class SAIMDataset(BaseDataset):

    def __init__(self):
        super(SAIMDataset, self).__init__()
        self.ms = 32
    
    def __getitem__(self, index):
        instance = {}
        # load instance_box information for trainning box
        image = Image.open(self.sampler[index]["image_path"])
        instance["PILimage"] = image
        instance["image"] = image
        w, h = image.size
        instance["image_width"] = w
        instance["image_height"] = h
        w = max(w//self.ms, 1)*self.ms
        h  =  max(h//self.ms, 1)*self.ms
        instance["origin_bbox"] = [ [b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in self.sampler[index]["bbox"]]
        instance["bbox"] =  [ [b[0], b[1], b[2], b[3]] for b in self.sampler[index]["bbox"]]
        instance["label"] = self.sampler[index]["label"]
        instance = transform(instance, size=(w, h), resize_only=False)
        instance["image_path"] = self.sampler[index]["image_path"].split(os.sep)[-1]
        
        return instance