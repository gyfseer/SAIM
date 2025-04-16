#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : saim_model.py
    @Time   : 2024/06/21 13:14:39
    @Authors : Jun Yang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
    @Description: write here
'''
import os
import torch
import shutil
import datetime

from tqdm import tqdm

from .state import State
from .model import Model
from .saim_dataloader import SAIMDataloader
from ..base_model import BaseModel
from ...data.database import MySQLDatabase

from .augment import transform

class SaimModel(BaseModel):

    def __init__(self):
        super(SaimModel, self).__init__()
        self.model_name = "saim"
        # 模型最大缩放尺度
        self.ms = 32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 声明模型
        self.model = Model(self.device)
        self.model.to(self.device)
        # 声明优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        # 初始化和检查环境变量
        self.init_and_check()

    def load(self):
        """
            恢复checkpoints
        """
        state = {}
        if os.path.exists(self.model_resume_from):
            try:
                checkpoint = torch.load(self.model_resume_from)
            except Exception as e:
                # 最近的模型已损坏，加载上一个版本的模型
                shutil.copyfile(src=self.model_resume_from_backup, dst=self.model_resume_from)
                checkpoint = torch.load(self.model_resume_from)
            
            # 加载预训练模型
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            state = checkpoint["state"]
        self.state = State(state)

    def train(self):
        self.state.set_name_mapping(MySQLDatabase.get_database_instance().select_category_map())
        self.dataloader = SAIMDataloader.get_dataloader()
        # 进入学习模式前, 初始化状态
        self.state.initialize(self.model, self.dataloader, self.device)
        # 收集损失
        loss_collections = [] 
        for instance in tqdm(self.dataloader, desc="update model"):
            self.state.update_state()
            loss = self.train_one(self.model, self.optimizer, self.state, instance, self.device)
            loss_collections.append(float(loss))
        # 记录损失
        if len(loss_collections):
            with open(self.loss_log, "a") as f:
                loss_str = f"time: {datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')} loss: {float(torch.mean(torch.tensor(loss_collections)))} "
                print(loss_str, file=f)
    
    def train_one(self, model, optimizer, state, instance, device):
        """
        训练模型

        Args:
            @Param model: 模型.
            @Param optimizer: 优化器.
            @Param state: 类内状态对象.
            @Param instance: 训练样本对象.
            @Param device: 训练设备.
        
        Returns:
            loss: 模型损失
        """
        images = []
        for item in instance:
            images.append(item["image"].unsqueeze(0))
        images = torch.cat(images, dim=0)
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = model(images, instance, state)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=100, norm_type=2)
        optimizer.step()
        return loss
    
    def detect(self, image):
        """
            模型检测
        
        Args:
            @Param image: 相机中获取的图片(PIL)
        
        Returns:
            返回 bboxes(边界框), labels(标签, 缺陷名), scores(得分)
        """
        bboxes, labels, scores = [], [], []
        if self.state.get_state_decode() is None:
            return bboxes, labels, scores
        if self.state.get_state_decode().__len__() == 0:
            return bboxes, labels, scores
        instance = self.preprocess(image)
        with torch.no_grad():
            bboxes, labels, scores = self.model(instance["image"].unsqueeze(0).to(self.device), instance=instance, state=self.state, train=False)
       
        labels = [ self.state.get_name_mapping()[label] for label in labels]    
       
        return bboxes, labels, scores

    def save(self):
        """
            保存模型checkpoins
        """
        checkpoint = {}
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        checkpoint["state"] = self.state.get_state()
        # 保存带状态的checkpoints
        if os.path.exists(self.model_resume_from):
            shutil.copyfile(src=self.model_resume_from, dst=self.model_resume_from_backup)
        torch.save(checkpoint, self.model_resume_from)
        # 保存不带状态的checkpoints
        checkpoint["state"] = {}
        if os.path.exists(self.model_resume_from_without_state):
            shutil.copyfile(src=self.model_resume_from_without_state, dst=self.model_resume_from_backup_without_state)
        torch.save(checkpoint, self.model_resume_from_without_state)
    
    def update_samples_score(self):
        # key: image id, value: 更新后的得分 
        new_score_info = []
        for instance in tqdm(self.dataloader, desc="update sample's score"):
            for item in instance:
                image = item["image"].unsqueeze(0).to(self.device, non_blocking=True)
                with torch.no_grad():
                    loss = self.model(image, [item], self.state)
                image_id = item["image_path"].split(".")[0]
                new_score_info.append((float(loss), image_id))
        super().update_samples_score(new_score_info, self.model_name)

    def init_and_check(self):
        """
            初始化参数配置
        """
        self.results_dir = "./results/SAIM/"
        # 模型保存目录
        self.weights_dir = self.results_dir + "weights/"
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        # 模型加载路径
        self.model_resume_from = self.weights_dir + "latest.pth"
        self.model_resume_from_backup = self.weights_dir + "backup.pth"
        self.model_resume_from_without_state = self.weights_dir + "latest_without_state.pth"
        self.model_resume_from_backup_without_state = self.weights_dir + "backup_without_state.pth"
        # 日志保存目录
        self.logs_dir = self.results_dir + "logs/"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        # 损失保存路径
        self.loss_log = self.logs_dir + "loss.txt"
    
    def preprocess(self, image):
        """
        Args:
            image:  PIL格式图片.

        Returns:
            instance: 数据增强后的图片.
        
        """
        instance = {}
        instance["image"] = image
        instance["PILimage"] = image
        w, h = image.size
        instance["image_width"] = w
        instance["image_height"] = h
        w = max(w//self.ms, 1)*self.ms
        h  = max(h//self.ms, 1)*self.ms
        instance = transform(instance, size=(w, h), add_noise=False)

        return instance