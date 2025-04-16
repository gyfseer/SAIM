#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : global_var.py
    @Time   : 2024/06/21 12:37:36
    @Authors : Jun Yang, Yifei Gong, Yiming Jiang
    @Version: 0.9
    @Contact: 
            Yang's email: 1830839103@qq.com;
           Gong's email: 315571669@qq.com;
           Jiang's email: 2773156542@qq.com
    @Description: write here
'''

import threading

class GlobalVar:
    """
        项目所用到的所有全局变量
    """

    dataloader_control_signal = threading.Event()
    current_model_name = None