#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : listener.py
    @Time   : 2023-03-04 18:11:57
    @Author : Stepend
    @Version: 0.5
    @Contact: stepend98@gmail.com
    @Description: event listen and handle.
'''

from PyQt5.Qt import *

import cv2
import time
import copy
import socket
import traceback
import numpy as np
from PIL import Image

from .const import *
from .common import *

from PyQt5 import QtGui
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QMessageBox

class DMThread(QThread):
    def __init__(self, parent):
        super(DMThread, self).__init__(parent)
        self.parent = parent
        self.image = None

    def run(self):
        time.sleep(1)
        add_log(self.parent.message_area_signal, "DM mode is running")
        while True:
            try:
                if self.parent.select_learn.isChecked() or self.parent.select_feedback.isChecked():
                    break
                self.parent.message.receiver_parser()
                self.parent.message.set_status(True)
                self.parent.message.get_sock().send(self.parent.message.tobyte())
                self.image = self.parent.message.get_sock().recv(self.parent.message.get_image_size(), socket.MSG_WAITALL)
                self.image = np.frombuffer(self.image, dtype=np.uint8)
                self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
                if self.image is None:
                    continue
                height, width, depth = self.image.shape
                self.image = QImage(self.image.data, width, height, width*depth, QImage.Format_RGB888)
                self.image = QtGui.QPixmap.fromImage(self.image)
                # # 展示到界面
                self.parent.image_area.canvas.update_list_label()
                self.parent.image_area.setCanvas(self.image)
            except Exception as e:
                traceback.print_exc()
                add_log(self.parent.message_area_signal, "Exception occurred while processing in 'DM' and restart 'DM'")
                break

        add_log(self.parent.message_area_signal, "DM mode is stopping")
        if self.parent.message.get_mode() == "FM":
            self.parent.feedback_mode_listener.start()


class FMThread(QThread):
    def __init__(self, parent):
        super(FMThread, self).__init__(parent)
        self.parent = parent
        self.image = None

    
    def run(self):
        time.sleep(1)
        add_log(self.parent.message_area_signal, "FM mode is running")
        while True:
            try:
                self.parent.message.receiver_parser()
                if self.parent.select_learn.isChecked() or self.parent.select_detect.isChecked():  
                    break  
                self.parent.message.set_status(True)
                self.parent.message.get_sock().send(self.parent.message.tobyte())
                self.image = self.parent.message.get_sock().recv(self.parent.message.get_image_size(), socket.MSG_WAITALL)
                self.image = np.frombuffer(self.image, dtype=np.uint8)
                self.image = cv2.imdecode(self.image, cv2.IMREAD_COLOR)
                if self.image is None:
                    continue
                height, width, depth = self.image.shape
                self.image = QImage(self.image.data, width, height, width*depth, QImage.Format_RGB888)
                self.image = QtGui.QPixmap.fromImage(self.image)
                # # 展示到界面
                self.parent.image_area.setCanvas(self.image)
                self.parent.image_area.canvas.update_list_label()
            except Exception as e:
                add_log(self.parent.message_area_signal, "Exception occurred while processing in 'FM' and restart 'FM'" )
                break

        add_log(self.parent.message_area_signal, "FM mode is stopping")
        if self.parent.message.get_mode() == "DM":
            self.parent.detect_mode_listener.start()
    


class CheckListener(QThread):

    def __init__(self, parent):
        super(CheckListener, self).__init__(parent)
        self.parent = parent
        self.db = parent.db
        self.cur_index = 0
        self.annotation = None
        self.image_id = None
        self.isQuit = False
        self.isDelete = False

    def run(self):
        try:
            results = self.db.select_image_all() # 所有图片的标注信息
            self.cur_index = len(results) - 1 # 从最新的开始遍历查看 
            for res in results:
                res[1]["label"] =  [ self.db.select_category_map()[label] for label in res[1]["label"]]
            while True:
                add_log(self.parent.message_area_signal, f"Checking id = {self.cur_index} of {len(results)} next: -> prev: <- update: ctrl+s delete: ctrl+d")
                self.cur_index = self.cur_index%len(results) if self.cur_index > 0 else (self.cur_index+len(results))%len(results)
                self.annotation = results[self.cur_index][1]
                self.image_id = results[self.cur_index][0]
                self.display()
                cur_index = self.cur_index
                while cur_index == self.cur_index:
                    if not self.parent.on_check or self.isQuit:
                        break
                    if self.isDelete:
                        results.pop(self.cur_index)
                        self.isDelete = False
                        break
                self.annotation = None
                if self.isQuit or not self.parent.on_check or len(results)==0:
                    break
        except Exception as e:
            traceback.print_exc()
            QMessageBox.information(self.parent, "connection error feedback", "check database configuration and password!")
        
        self.annotation = None
        self.parent.on_check = False
        self.isQuit = False
        self.parent.check_btn.setStyleSheet(BUTTON)
    
    def nxt(self):
            self.cur_index += 1
    
    def prev(self):
            self.cur_index -= 1
    
    def quit(self):
            self.isQuit = True
    
    def update(self):
        # 先删除数据库内容
        self.db.delete_instance_by_image_id(self.image_id)
        self.annotation["bbox"] = self.parent.message.get_bboxes()
        self.annotation["label"] = self.parent.message.get_labels()
        image = Image.open(self.db.get_image_root_dir()+str(self.image_id)+".jpg")
        self.db.insert_instance(copy.deepcopy(self.annotation), image)

    def display(self):
        self.parent.message.set_bboxes(self.annotation["bbox"])
        self.parent.message.set_labels(self.annotation["label"])
        self.parent.image_area.canvas.update_list_label()
        self.parent.image_area.setCanvas(QPixmap(self.db.get_image_root_dir()+str(self.image_id)+".jpg"))
    
    def delete(self):
        """Delete instance whose name is name
        
        """
        if self.db.delete_instance_by_image_id(self.image_id):
            self.isDelete = True
    

