# -*- encoding: utf-8 -*-
'''
    @文件名称   : client.py
    @创建时间   : 2023/12/28 16:22:00
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 图形用户界面主程序
    @参考地址   : 无
'''
import sys

from PyQt5.Qt import *
from .gui.mainwindow import MainWindow
from .data.database import MySQLDatabase

__appname__ = "SDDA - UI"

def main():
    # sys.argv: command line arguments
   app = QApplication(sys.argv)

   # create the main window
   window = MainWindow(MySQLDatabase.get_database_instance())

   # set window title
   window.setWindowTitle(__appname__)

   # show the main window
   window.show()

   # execute the application and run into message queue
   sys.exit(app.exec_())
