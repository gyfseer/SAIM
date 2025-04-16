import json
from .const import *

class Message:

    def __init__(self, sock=None, bboxes=[], labels=[], scores=[], mode="LM", connect="disconnect", status=False):
        self.sock = sock
        self.bboxes = bboxes
        self.labels = labels
        self.scores = scores
        self.mode = mode
        self.connect = connect
        self.status = status
        self.message = None
        self.image_size = None
        self.isSave = False
    
    def tobyte(self):
        return bytes(
            json.dumps(
                {
                    "label": self.labels,
                    "bbox": self.bboxes,
                    "mode": self.mode,
                    "connect": self.connect,
                    "status": self.status,
                    "isSave": self.isSave
                }
            ).encode("utf-8")
        )
    def tostring(self):
        return f"label: {self.labels}, bbox: {self.bboxes}, mode: {self.mode}, connect: {self.connect}, status: {self.status}, isSave: {self.isSave}"


    def receiver_parser(self):
        self.server_msg = json.loads(self.sock.recv(MAX_RECEIVE_LENGTH))
        # self.mode = self.server_msg["mode"]
        self.labels = self.server_msg["label"]
        self.bboxes = self.server_msg["bbox"]
        self.scores = self.server_msg["score"]
        self.image_size = self.server_msg["image_size"]
        self.message = self.server_msg["message"]
    
    def get_message(self):
        return self.message
    
    def get_image_size(self):
        return self.image_size


    def set_sock(self, sock):
        self.sock = sock
    
    def get_sock(self):
        return self.sock

    def set_bboxes(self, bboxes):
        self.bboxes = bboxes

    def get_bboxes(self):
        return self.bboxes

    def set_labels(self, labels):
        self.labels = labels
    
    def get_labels(self):
        return self.labels
    
    def get_scores(self):
        return self.scores

    def set_mode(self, mode):
        self.mode = mode

    def get_mode(self):
        return self.mode

    def set_connect(self, connect):
        self.connect = connect
    
    def get_connect(self):
        return self.connect
    
    def set_status(self, status):
        self.status = status
    
    def get_status(self):
        return self.status
    
    def set_isSave(self, isSave):
        self.isSave = isSave