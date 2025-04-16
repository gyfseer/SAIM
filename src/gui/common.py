#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
    @File   : common.py
    @Time   : 2023-03-04 14:19:57
    @Author : Stepend
    @Version: 0.5
    @Contact: stepend98@gmail.com
    @Description: define some common use function here.
'''

import time
import socket
import logging
import datetime
from PyQt5.Qt import *
from .const import *

def connect_to_server(parent, ip, port, button, radio_button, message):
    """Click button to connect to the server.

    Args:
        parent: the parent object.
        ip: server's ip.
        port: server's port.
        button: button object.
        radio_button: select mode.
        message: an object is used to manage data of client and server.
    
    """
    try:
        if message.get_sock() is None:
            # create TCP socket
            message.set_sock(
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            )
            message.get_sock().connect((ip, port))
            message.set_connect(CONNECT)
            button.setStyleSheet(BUTTON_ACTIVE)
            message.receiver_parser()
            radio_button.setChecked(True)
            add_log(parent.message_area_signal, message.get_message())
        else:
            # disconnect from server
            message.set_connect(DISCONNECT)
            message.get_sock().send(
                message.tobyte()
            )
            message.get_sock().close()
            message.set_sock(None)
            button.setStyleSheet(BUTTON)
            add_log(parent.message_area_signal, "Client Disconnected")

    except Exception as e:
        add_log(parent.message_area_signal, "connection error feedback, check IP and Port")
        QMessageBox.information(parent, "connection error feedback", "check IP and Port")
        message.set_sock(None)

def check_data(parent, button, message, check_listener):
    if message.get_sock() is None:
        if not parent.on_check:
            parent.on_check = not parent.on_check
            button.setStyleSheet(BUTTON_ACTIVE)
            # do things
            check_listener.start()
        else:
            parent.on_check = not parent.on_check
            button.setStyleSheet(BUTTON)
    else:
        QMessageBox.information(parent, "check error feedback", "client is connected to server, please disconnect and retry.")


def get_mode(listQRB):
    """Check which radio button is selected.
    
    Args:
        listQRB: list of Radio button.
    """
    for QRB in listQRB:
        if QRB.isChecked():
            return QRB.text()

def change_mode(listQRB, message, detect_mode_listener, feedback_mode_listener):
    """Select radio button to switch mode.

    Args:
        listQRB: list of Radio button.
        message: message object to manage data of client.
        detect_mode_listener: handle events between client and server in mode of 'DM'. (a QThread)
        feedback_mode_listener: handle events between client and server in mode of 'FM'. (a QThread)
    
    """
    if message.get_sock() is None:
        return

    mode = get_mode(listQRB)
    if mode == message.get_mode():
        return
    message.set_bboxes([])
    message.set_labels([])
    message.set_isSave(False)
    if message.get_mode()=="LM":
        message.set_mode(mode)
        if mode == "DM" and message.get_sock() is not None:
            detect_mode_listener.start()
        if mode == "FM" and message.get_sock() is not None:
            feedback_mode_listener.start()
        message.get_sock().send(
            message.tobyte()
        )
        time.sleep(1) 
    
    if message.get_mode()=="DM":
        message.set_mode(mode)

    if message.get_mode()=="FM":
        pass

def isSave(message, isSave):
    message.set_isSave(isSave)


def send_feedback(message, listQRB, parent):
    """Send message to server.
    
    Args:
        message: message object to manage data of client.
        listQRB: list of Radio button.
        parent: component's parent object.

    """
    time.sleep(1)
    mode = get_mode(listQRB)
    if mode != "FM" and message.get_mode() != "FM":
        return
    if message.get_sock() is None:
        QMessageBox.information(parent, "connection error feedback", "check the connection")
        return
    message.set_mode(mode)
    message.get_sock().send(
        message.tobyte()
    )
    add_log(parent.message_area_signal, "Feedback a message successfully")
    message.set_bboxes([])
    message.set_labels([])
    message.set_isSave(False)

def get_pressed(button):
    button.setStyleSheet(BUTTON_ACTIVE)

def get_released(button):
    button.setStyleSheet(BUTTON)

def date_string(s):
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{date}: {s}"

def add_log(o, s):
    """Add a log.

    Args:
        o: a signal object.
        s: a string.
    
    """
    logging.info(s)
    o.signal.emit(date_string(s))

class MessageAreaSignal(QObject):

    signal = pyqtSignal(str)