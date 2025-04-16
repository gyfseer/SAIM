# -*- encoding: utf-8 -*-
'''
    @文件名称   : message.py
    @创建时间   : 2023/12/28 17:25:18
    @作者名称   : Stepend
    @当前版本   : 1.0
    @联系方式   : stepend98@gmail.com
    @文件描述   : 服务端信息收集
    @参考地址   : 无
'''
import cv2
import json
import numpy as np

from .mode import Mode
class Message:
    """Summarize class here.
    
    Manage the message information of between client and server.

    Attributes:
        sock: client request socket.
        addr: client request address.
        mode: working mode.
        client_pkg: client package.
    """

    def __init__(self, sock=None, addr=None, mode=Mode.LM):
        self.mode = mode
        self.sock = sock
        self.addr = addr
        self.connect = False
        self.status = False

        self.label = []
        self.bbox = []
        self.score = []
        self.image_size = None
        self.isSave = False

        self.message = None
        self.text_message = None
        self.image_message = None

        self.MAX_RECEIVE_LENGTH = 65535 # Maximum length of a message from client.
    
    def tobytes(self):
        return bytes(
            json.dumps(
                {
                    "mode": self.mode,
                    "label": self.label,
                    "bbox": [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in self.bbox],
                    "score": self.score,
                    "image_size":self.image_size,
                    "message": self.message

                }
            ).encode("utf-8")
        )

    def receiver_parser(self):
        """Receive a message from client."""
        client_msg = json.loads(self.sock.recv(self.MAX_RECEIVE_LENGTH))
        self.label = client_msg["label"]
        self.bbox = client_msg["bbox"]
        self.mode = client_msg["mode"]
        self.connect = client_msg["connect"]
        self.status = client_msg["status"]
        self.isSave = client_msg["isSave"]

    
    def instance_parser(self):
        """Parser instance from received message
        
        """
        instance = {}
        instance["bbox"] = self.bbox
        instance["label"] = self.label

        return instance
    

    def prepare_server_pkg(self, image, bbox, label, score):
        """Prepare server package

        Args:
            image: PIL image.
            bbox: predict bounding box.
            label: predict label.
            score: predict score.
        
        """
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _ ,image = cv2.imencode(".jpg", image)
        self.image_message = image.tobytes()

        self.score = score
        self.label = label
        self.bbox = bbox
        self.image_size = len(self.image_message)

        self.text_message = self.tobytes()
    
    def send_message(self, **kwargs):
        self.text_message = self.tobytes()
        self.sock.send(self.text_message)
        self.score = []
        self.label = []
        self.bbox = []
    
    def send(self):
        self.send_message()
        self.receiver_parser()
        self.send_image()
    
    def get_label(self):
        return self.label
    
    def get_bbox(self):
        return self.bbox
    
    def send_image(self):
        self.sock.send(self.image_message)
    
    def get_connect(self):
        return self.connect
    
    def set_connect(self, connect):
        self.connect = connect

    def get_status(self):
        return self.status
    
    def set_status(self, status):
        self.status = status

    def set_sock(self, sock):
        self.sock = sock
    
    def get_sock(self):
        return self.sock
    
    def set_addr(self, addr):
        self.addr = addr
    def get_addr(self):
        return self.addr

    def set_mode(self, mode):
        self.mode = mode
    def get_mode(self):
        return self.mode
    
    def set_message(self, message):
        self.message = message

    def get_message(self):
        return self.message
    
    def get_isSave(self):
        return self.isSave