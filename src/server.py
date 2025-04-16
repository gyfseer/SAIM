#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : server.py
    @Time   : 2024/06/20 14:35:21
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''

import time
import torch
import socket
import logging
import traceback
import threading
import os

from PIL import Image, ImageDraw, ImageFont
from .utils.mode import Mode
from .utils.message import Message
from .global_var import GlobalVar
from .data.database import MySQLDatabase
from .model_hub.model_hub import ModelHub

JSON_PATH = ""
IMAGE_PATH = "datasets/insert"


class Server:

    def __init__(self, args):
        # 用于通信的信息类.
        self.msg = Message()
        # 指定计算设备
        self.compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 加载数据集

        # 加载模型
        self.model = ModelHub.load_model(args.model_name)
        self.model.load()

        # 加载数据集
        MySQLDatabase.get_database_instance().check_score_column(args.model_name)
        if JSON_PATH != "":
            MySQLDatabase.get_database_instance().insert_coco_directory(IMAGE_PATH, JSON_PATH)

        # 创建线程监听请求
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', 9999))
        self.server.listen(1)  # 最大监听数.
        self.listen_thread = threading.Thread(target=Server.request_listener, args=[self.server, self.msg])
        self.listen_thread.daemon = True
        self.listen_thread.start()

        logging.info("Server On")

    def run(self):
        while True:
            if self.msg.get_mode() == Mode.LM:
                self.run_LM()

            if self.msg.get_mode() == Mode.DM:
                self.run_DM()

            # TODO: 推理已有数据
            # self.detect_database()
            time.sleep(1)

    def run_LM(self):
        """
            训练模式: 自动加载train目录中的所有图片
        """
        # 新建一个线程监听客户端的请求
        recv_thread = threading.Thread(target=Server.msg_receiver, args=[self.msg])
        recv_thread.start()

        GlobalVar.dataloader_control_signal.set()
        # 学习模式主循环
        while True:
            self.model.train()
            # 保存模型
            self.model.save()
            # 更新样本得分
            self.model.update_samples_score()
            # 如果客户端改变模式, 那么中断训练
            if self.msg.get_mode() != Mode.LM:
                break
        logging.info("Step out 'LM' mode")

    def run_DM(self):
        """
            检测模式
        """
        logging.info("Step into 'DM' mode")

        # 指定图片目录
        image_dir = "datasets/val"
        output_dir = "results/SAIM/images"

        # 获取目录下所有图片文件
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print("get all the image down\n")
        while True:
            for img_name in image_files:
                img_path = os.path.join(image_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')  # 使用PIL读取图像
                except Exception as e:
                    logging.warning(f"Failed to open image {img_name}: {e}")
                    continue
                
                # 检测图像
                pred_bbox, label, score = self.model.detect(image)
                # 绘制检测结果
                if pred_bbox and label and score:
                    print("successfully get the pred_bbox\n")
                    draw = ImageDraw.Draw(image)
                    for box, lbl, scr in zip(pred_bbox, label, score):
                        if len(box) != 4:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-BoldOblique.ttf", 20)
                        text = f"{lbl}: {scr:.2f}"
                        draw.text((x1, max(y1 - 12, 0)), text, font=font, fill="red")
                    # 保存图像
                    save_path = os.path.join(output_dir, img_name)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    image.save(save_path)
                    print("successfully save the image\n")
            print("finish\n")
            self.msg.mode=Mode.LM
            break
        
        logging.info("Step out 'DM' mode")

    @staticmethod
    def msg_receiver(msg):
        """
        接收客户端的消息

        Args:
            msg: 来自客户端的消息
        """
        mode = msg.get_mode()
        logging.info(f"{mode} receiver is running")
        while True:
            try:
                if msg.get_addr() is not None and msg.get_sock() is not None:
                    # 已经建立连接
                    msg.receiver_parser()
                # 断开连接, 改变保存状态
                if not msg.get_connect():
                    msg.set_sock(None)
                    msg.set_addr(None)
                # 客户端改变状态
                if msg.get_mode() != mode:
                    GlobalVar.dataloader_control_signal.clear()
                    break
            except Exception:
                msg.set_addr(None)
                msg.set_sock(None)
            time.sleep(0.1)

        logging.info(f"{mode} receiver is over")

    @staticmethod
    def request_listener(server, msg):
        """
        监听客户端连接请求

        Args: 
            server: 服务器.
            msg: 客户端消息.
        
        """
        while True:
            if msg.get_addr() is None and msg.get_sock() is None:
                sock, addr = server.accept()
                msg.set_addr(addr)
                msg.set_sock(sock)
                msg.set_message(f"welcome to connect to SDDA server and you can interact with SDDA server now!")
                msg.send_message()
                logging.info(f"accepted connection from: {addr}")
            time.sleep(0.1)
