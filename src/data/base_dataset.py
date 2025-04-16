#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : base_dataset.py
    @Time   : 2024/06/20 14:37:48
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''
import threading
import numpy as np
from torch.utils.data import Dataset
from ..global_var import GlobalVar

from .database import *

class BaseDataset(Dataset):
    """
        基础数据集类(数据增强由具体实现决定)
    
    Attributes:
        number: 单类每轮采样次数 default: 64
        db: 数据库对象
    
    Methods:
         prob_based_sample: 依概率采样原始数据
    """

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.number = 64
        self.db = MySQLDatabase.get_database_instance()
    
    def prob_based_sample(self):
        """
            依概率采样原始数据
                    
        Returns:
            return_description
        """
        # 原始数据集中采样得到用于训练的采样数据集
        self.sampler = []
        category_ids = self.db.select_category_ids()
        for category_id in category_ids:
            image_ids, scores = self.db.select_image_id_prob_by_category_id(category_id)
            if len(scores) == 0:
                continue
            scores = np.array(scores)
            # TODO: max or clip?
            scores -= np.max(scores)
            # p表示得分概率化后的值
            p = np.exp(scores)/np.sum(np.exp(scores))
            # 采样
            indexes = np.random.choice(np.arange(0, len(image_ids)), self.number, p=p)
            image_ids = [image_ids[x] for x in indexes]
            # 根据图片id查数据
            results = self.db.select_image_by_id(image_ids)
            for res in results:
                instance = {}
                annos = json.loads(res[1])
                instance["image_path"] = IMAGE_ROOT_DIR + str(res[0]) + ".jpg"
                instance["label"] = annos["label"]
                instance["bbox"] = annos["bbox"]
                self.sampler.append(instance)                

        # 更新任务
        thread = threading.Thread(target=self.update_thread)
        thread.start()      

    def update_thread(self):
        category_ids = self.db.select_category_ids()
        for category_id in category_ids:
            self.db.upate_score_by_category_id(category_id, self.number, GlobalVar.current_model_name)
    
    def __len__(self):
        return self.sampler.__len__()
    
    def __getitem__(self, index):
        raise NotImplementedError("Customed class dataset must implement function __getitem__")
