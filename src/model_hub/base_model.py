#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : base_model.py
    @Time   : 2024/06/20 16:47:44
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''

from ..data.database import MySQLDatabase

class BaseModel:
    """
        基础模型, 所有模型实现均需要继承该模型
    
    Methods:
        train(): 模型训练
        detect(): 模型检测
        update_samples_score(): 更新数据库得分
    """

    def load(self):
        """
            加载模型
        """
        raise NotImplementedError("BaseModel.load() is not implemented")

    def train(self):
        """
            训练模型
        """
        raise NotImplementedError("BaseModel.train() is not implemented")

    def detect(self, image):
        """
            模型检测

        Args:
            @Param image: 相机中获取的图片(PIL)        
        
        Returns:
            返回 bboxes(边界框), labels(标签, 缺陷名), scores(得分)
        """
        raise NotImplementedError("BaseModel.detect() is not implemented")

    def save(self):
        """
            模型检测
        
        Returns:
            返回 bboxes(边界框), labels(标签, 缺陷名), scores(得分)
        """
        raise NotImplementedError("BaseModel.save() is not implemented")


    def update_samples_score(self):
        """
            更新样本数据得分
        """
        raise NotImplementedError("BaseModel.update_samples_score() is not implemented")
    
    def update_samples_score(self, new_score_info,  model_name):
        """
            更新样本数据得分
            TODO: 待优化, 优化为batch插入

        Args:
            @Param new_scores_info: 更新后的得分信息. eg:
                                 new_score_info = [
                                    (image_id_1, score_1),
                                    (image_id_2, score_2),
                                                  ..............
                                    (image_id_n, score_n),
                                 ]       
            @Param model_name: 模型名
        """
        # 更新     
        MySQLDatabase.get_database_instance().update_scores_by_modelname(new_score_info, model_name)