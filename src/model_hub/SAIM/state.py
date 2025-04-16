# -*- encoding: utf-8 -*-
'''
    @文件名称   : state.py
    @创建时间   : 2023/12/28 11:44:27
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 类别状态管理
    @参考地址   : 无
'''
import torch
import random
from tqdm import tqdm

class State:
    """

    管理类别状态的类. 学习过程中, 给每个类分配一个聚类中心, 通过自适应迭代更新均匀分布在单位球面上.
    
    Attributes:
        state : 字典, key为类别名, value为N维向量. (1, N)
        state_matrix: 由state的类别向量拼接成的矩阵.
        state_encode: key为类别名, value为state_matrix下标.
        state_decode: key为state_matrix下标, value为类别名.
    """
    def __init__(self, state):
        self.state = state
        # key: 标签id  value: 标签名
        self.name_mapping = None
        if len(self.state):
            self.build_state_matrix_and_encode()
            self.build_state_decode()
        else:
            self.state_matrix = None 
            self.state_encode = None
            self.state_decode = None
        self.step = 16
        self.err = {2: 6.017008316290887e-3, 3: 0.007694659288972616, 4: 0.0071899626115337014, 5: 0.007132688881829381, 6: 0.00736707166954875, 7: 0.007862207781523466, 8: 0.007292385678738356, 9: 0.008632712997496128, 10: 0.00731934979557991, 11: 0.008643737062811852, 12: 0.008328444324433804, 13: 0.015951938927173615, 14: 0.014349878765642643, 15: 0.01318759098649025, 16: 0.02130085788667202, 17: 0.015615168958902359, 18: 0.020766962319612503, 19: 0.022458994761109352, 20: 0.021684257313609123, 21: 0.027185047045350075, 22: 0.038083724677562714, 23: 0.029705462977290154, 24: 0.025738265365362167}

    def initialize(self, model, dataloader, device):
        """
        
        初始化新标签类别状态

        Args:
            model: 深度模型.
            dataloader: 训练的数据加载器.
            device: 计算设备gpu 或 cpu.

        """
        self.init_state(model, dataloader, device)
        self.build_state_matrix_and_encode()
        self.build_state_decode()
    
    @staticmethod
    def norm(x):
        """
            单位化状态表征
        """
        return x/torch.sqrt(torch.sum(x*x,dim=1)).view(-1,1)

    def init_state(self, model, dataloader, device):
        """
        初始化状态表征

        Args:
            model: 深度模型.
            dataloader: 训练的数据加载器.
            device: 计算设备gpu 或 cpu.
        """
                
        temp_state = {}
        for batch_instance in tqdm(dataloader, desc="update state"):

            for instance in batch_instance:
                # 如果没有标签，那么不用来添加状态
                if len(instance["label"]) == 0:
                    continue
                # 如果标签中没有新的状态，那么不用用来获取初始状态
                if not self.has_new_label(instance["label"]) or self.name_mapping[instance["label"][0]].endswith("background"):
                    continue
                sorted_bboxes, sorted_labels = State.sort_bboxes_by_area(instance["bbox"], instance["label"])
                image = instance["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(image, train=False, init_state=True)
                mask = torch.zeros_like(out[0, 0, :, :]).to(device)
                for index, bbox in enumerate(sorted_bboxes):
                    name = sorted_labels[index]
                    if name in self.state:
                        continue
                    cy, cx = State.search_xy_in_mask(bbox, mask, self.step)
                    if cy == -1 and cx == -1:
                        continue
                    mask[cy, cx] = 2
                    try:
                        temp_state[name] = temp_state[name] + out[:1, 5:, cy, cx]
                    except KeyError:
                        temp_state[name] = out[:1, 5:, cy, cx]
                # 添加每个任务的背景类
                # background_name = sorted_labels[0].split("_")[0] + "_"
                # if sorted_labels[0].split("_")[0] not in self.state:
                
        for name in temp_state:
            self.state[name] = State.norm(temp_state[name])

    def has_new_label(self, label_list):
        for label in label_list:
            if label not in self.state:
                return True
        return False      

    @staticmethod
    def sort_bboxes_by_area(bboxes, labels):
        # 计算每个矩形的面积
        areas = [abs((x2 - x1) * (y2 - y1)) for x1, y1, x2, y2 in bboxes]

        # 将(面积,索引)元组列表排序,获得索引列表 
        sorted_idx = [i[0] for i in sorted(enumerate(areas), key=lambda x:x[1])]

        # 使用索引列表对bboxes和labels同时排序
        sorted_bboxes = [bboxes[i] for i in sorted_idx]
        sorted_labels = [labels[i] for i in sorted_idx]

        return sorted_bboxes, sorted_labels

    @staticmethod
    def search_xy_in_mask(bbox, mask, step):
        l, t, r, b = bbox
        cx = round((l+r)/2/step)
        cy = round((t+b)/2/step)
        h, w = mask.shape
        cy = cy if cy < h else cy-1
        cx = cx if cx < w else cx-1
        if mask[cy, cx] != 2:
            return cy, cx
        # 矩形框中心重叠
        max_edge = max(r-l, b-t)
        delta_x = (r-l)/max_edge
        delta_y = (b-t)/max_edge
        s = 1
        while True :
            # 沿左上搜索
            x = round(cx-s*delta_x)
            y = round(cy-s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break 
            # 沿右上搜索
            x = round(cx+s*delta_x)
            y = round(cy-s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            # 沿左下搜索
            x = round(cx-s*delta_x)
            y = round(cy+s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            # 沿右下搜索
            x = round(cx+s*delta_x)
            y = round(cy+s*delta_y)
            if x>=0 and x<w and y>=0 and y<h:
                if mask[y, x] != 2:
                    cx, cy = x, y
                    break
            s += 1
            if s == 100:
                cy = -1
                cx = -1
                break
        return cy, cx   

    def build_state_matrix_and_encode(self):
        """
        由state构造矩阵 以及建立 类别名和矩阵下标的映射
        """
        index = 0
        collections = []
        self.state_encode = {}
        for name in self.state:
            collections.append(self.state[name])
            self.state_encode[name] = index
            index += 1
        try:
            self.state_matrix = torch.cat(collections, dim=0)
        except Exception:
            self.state_matrix = None
    
    def build_state_decode(self):
        """
        构造矩阵下标和类别名的映射
        """
        self.state_decode = {}
        for name in self.state_encode:
            self.state_decode[self.state_encode[name]]=name

    def update_state(self):
        """
        更新锚点状态, 调整类间距, 原理最近异类锚点.
        当状态表征均匀分布在单位球上, 即状态表征和向量的模小于给定值, 那么不更新状态.
        否则距离最近的两个向量存在斥力,相互远离,每次远离距离不过1度.
        """
        if len(self.state) < 2:
            return 
        cov_matrix = self.state_matrix @ self.state_matrix.T
        n = cov_matrix.shape[0]
        # 方差最小化，为了并行运算找到最近异类锚点
        cov_matrix[range(n), range(n)] = -2
        values, indices = torch.max(cov_matrix, dim=1)
        cov_matrix[range(n), range(n)] = 2
        min_values, min_indices = torch.min(cov_matrix, dim=1)
        new_state = []
        lr = 1e-3
        for i, j in enumerate(indices):
            gap_ij = self.state_matrix[i:i+1, :] - self.state_matrix[j:j+1, :]
            # 如果在1°以内, 我们认为两个向量是重合的,
            if torch.sum(gap_ij*gap_ij) < 0.0175: 
                rand_mov = torch.zeros_like(self.state_matrix[j:j+1, :])
                rand_mov[0, random.randint(0, rand_mov.shape[1])-1] = 0.034
                new_state.append(self.state_matrix[i:i+1, :] + rand_mov)
            else:
                new_state.append(self.state_matrix[i:i+1, :] + lr *(random.random()+1)/2 *gap_ij)
        new_state = torch.cat(new_state, dim=0)

        # 如果已经均匀，那么不更新
        if torch.sum(torch.sum(self.state_matrix, dim=0) * torch.sum(self.state_matrix, dim=0)) <= torch.sum(torch.sum(new_state, dim=0) * torch.sum(new_state, dim=0)) and torch.sum(torch.sum(self.state_matrix, dim=0) * torch.sum(self.state_matrix, dim=0)) < self.err[len(self.state)]:
            return
        print(f"update state and sum is {torch.sum(torch.sum(self.state_matrix, dim=0) * torch.sum(self.state_matrix, dim=0))}")
        self.state_matrix = new_state
        self.state_matrix = State.norm(self.state_matrix)

        # 更新状态，状态编码器，状态解码器
        for name in self.state:
            self.state[name] = self.state_matrix[self.state_encode[name], :].view(1, -1)
        
        self.build_state_matrix_and_encode()
        self.build_state_decode()

    def get_state_matrix(self):
        return self.state_matrix

    def get_state_encode(self):
        return self.state_encode
    
    def get_state_decode(self):
        return self.state_decode

    def get_state(self):
        return self.state

    def set_name_mapping(self, name_mapping):
        self.name_mapping = name_mapping
    
    def get_name_mapping(self):
        return self.name_mapping