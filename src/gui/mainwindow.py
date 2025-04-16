# # -*- encoding: utf-8 -*-
# '''
#     @文件名称   : mainwindow.py
#     @创建时间   : 2023/12/28 16:27:43
#     @作者名称   : Stepend
#     @当前版本   : 1.0
#     @联系方式   : stepend98@gmail.com
#     @文件描述   : 主界面程序
#     @参考地址   : 无
# '''
# from PyQt5.Qt import *
# from PyQt5 import QtGui

# from .const import *
# from .common import *
# from .listener import *
# from .canvas import Canvas
# from .message import Message

# class MainWindow(QWidget):
    
#     def __init__(self, db):
#         super(MainWindow, self).__init__()
#         self.db = db
#         self.setObjectName("MainWindow")
#         self.message = Message()
#         self.setMinimumWidth(1280)
#         self.setMinimumHeight(800)
#         # label list area
#         self.label_area_title = QLabel("Label Area")
#         self.label_area = QListWidget(self)
#         self.label_area.setStyleSheet(LABEL_AREA_STYLE)
#         self.save_button = QPushButton("Save")
#         self.save_button.setStyleSheet(BUTTON)
#         self.unsave_button = QPushButton("Unsave")
#         self.unsave_button.setStyleSheet(BUTTON)
#         self.save_button_layout = QHBoxLayout()
#         self.save_button_layout.addWidget(self.save_button)
#         self.save_button_layout.addWidget(self.unsave_button)
#         self.save_button_window = QWidget()
#         self.save_button_window.setLayout(self.save_button_layout)

#         self.label_button = QPushButton("Clear ALL")
#         self.label_button.setStyleSheet(BUTTON)
#         self.feedback_button = QPushButton("Feedback")
#         self.feedback_button.setStyleSheet(BUTTON)
#         self.label_button_layout = QVBoxLayout()
#         self.label_button_layout.addWidget(self.save_button_window)
#         self.label_button_layout.addWidget(self.label_button)
#         self.label_button_layout.addWidget(self.feedback_button)
#         self.label_button_window = QWidget()
#         self.label_button_window.setLayout(self.label_button_layout)

#         self.label_area_layout = QVBoxLayout()
#         self.label_area_layout.addWidget(self.label_area_title, 1)
#         self.label_area_layout.addWidget(self.label_area, 7)
#         self.label_area_layout.addWidget(self.label_button_window, 2)
#         self.label_area_window = QWidget()
#         self.label_area_window.setLayout(self.label_area_layout)

#         # image area
#         self.image_area_title = QLabel("Image Area")
#         # 从数据库中获取标签列表
#         self.image_area = Canvas(self, self.label_area, message=self.message, list_item=self.db.select_category_names(), origin=True)
#         self.image_area.setStyleSheet(IMAGE_AREA_STYLE)
#         self.image_area_layout = QVBoxLayout()
#         self.image_area_layout.addWidget(self.image_area_title, 1)
#         self.image_area_layout.addWidget(self.image_area, 9)
#         self.image_area_window = QWidget()
#         self.image_area_window.setLayout(self.image_area_layout)


#         # message area
#         self.message_area_title = QLabel("Message Area")
#         self.message_area = QPlainTextEdit(self)
#         self.message_area.setStyleSheet(MESSAGE_AREA_STYLE)
#         self.message_area_layout = QVBoxLayout()
#         self.message_area_layout.addWidget(self.message_area_title, 1)
#         self.message_area_layout.addWidget(self.message_area, 9)
#         self.message_area_window = QWidget()
#         self.message_area_window.setLayout(self.message_area_layout)
#         self.message_area_window.setStyleSheet(ADD_MARGIN)
#         self.message_area_signal = MessageAreaSignal()

#         # function area
#         # IP input line 
#         self.ip_label = QLabel("I   P:", self)
#         self.ip_input = QLineEdit(self)
#         self.ip_input.setStyleSheet(BORDER)
#         self.ip_layout = QHBoxLayout()
#         self.ip_layout.addWidget(self.ip_label)
#         self.ip_layout.addWidget(self.ip_input)
#         self.ip_window = QWidget()
#         self.ip_window.setLayout(self.ip_layout)
#         # Port input line
#         self.port_label = QLabel("Port:", self)
#         self.port_input = QLineEdit(self)
#         self.port_input.setStyleSheet(BORDER)
#         self.port_layout = QHBoxLayout()
#         self.port_layout.addWidget(self.port_label)
#         self.port_layout.addWidget(self.port_input)
#         self.port_window = QWidget()
#         self.port_window.setLayout(self.port_layout)
        
#         # connect socket button
#         self.connect_btn = QPushButton("Connect")
#         self.connect_btn.setStyleSheet(BUTTON)
#         # set socket connection layout
#         self.connect_layout = QVBoxLayout()
#         self.connect_layout.addWidget(self.ip_window)
#         self.connect_layout.addWidget(self.port_window)
#         self.connect_layout.addWidget(self.connect_btn)
#         # add layout to container
#         self.connect_window = QWidget()
#         self.connect_window.setLayout(self.connect_layout)


#         # mode select list [detect mode, supervise mode, learn mode], radio button
#         # define component
#         self.listQRB = []
#         self.select_detect = QRadioButton("DM")
#         self.select_detect.setStyleSheet(BORDER)
#         self.listQRB.append(self.select_detect)
#         # self.select_feedback = QRadioButton("FM")
#         # self.select_feedback.setStyleSheet(BORDER)
#         # self.listQRB.append(self.select_feedback)
#         self.select_learn = QRadioButton("LM")
#         self.select_learn.setStyleSheet(BORDER)
#         self.listQRB.append(self.select_learn)

#         # set layout
#         self.select_mode_layout = QHBoxLayout()
#         self.select_mode_layout.addWidget(self.select_detect)
#         # self.select_mode_layout.addWidget(self.select_feedback)
#         self.select_mode_layout.addWidget(self.select_learn)

#         # add layout to container
#         self.select_mode_window = QWidget()
#         self.select_mode_window.setLayout(self.select_mode_layout)

#         #TODO: a check button to check data of database
#         self.check_btn = QPushButton("Check")
#         self.check_btn.setStyleSheet(BUTTON)
#         self.check_btn_layout = QVBoxLayout()
#         self.check_btn_layout.addWidget(self.check_btn)
#         self.check_btn_window = QWidget()
#         self.check_btn_window.setLayout(self.check_btn_layout)
#         self.on_check = False

#         # set function_column layout
#         self.function_area_layout = QVBoxLayout()
#         self.function_area_layout.addWidget(self.connect_window)
#         self.function_area_layout.addWidget(self.select_mode_window)
#         self.function_area_layout.addWidget(self.check_btn_window)
#         self.function_area_window = QWidget()
#         self.function_area_window.setLayout(self.function_area_layout)
#         self.function_area_window.setStyleSheet(FUNCTION_COLUMN_STYLE)

#         self.function_area_title = QLabel("Function Area")
#         self.function_area_layout_with_title = QVBoxLayout()
#         self.function_area_layout_with_title.addWidget(self.function_area_title, 1)
#         self.function_area_layout_with_title.addWidget(self.function_area_window, 9)
#         self.function_area_window_with_title = QWidget()
#         self.function_area_window_with_title.setLayout(self.function_area_layout_with_title)

#         # add layout on image and label
#         self.image_label_layout = QHBoxLayout()
#         self.image_label_layout.addWidget(self.function_area_window_with_title, 2)
#         self.image_label_layout.addWidget(self.image_area_window, 6)
#         self.image_label_layout.addWidget(self.label_area_window, 2)
#         self.image_label_window = QWidget(self)
#         self.image_label_window.setLayout(self.image_label_layout)

#         self.display_layout = QVBoxLayout()
#         self.display_layout.addWidget(self.image_label_window, 8)
#         self.display_layout.addWidget(self.message_area_window, 2)
        
#         # # add layout to the container
#         self.setLayout(self.display_layout)
#         # set main window style
#         self.setStyleSheet(MAIN_WINDOW_STYLE)


#         self.detect_mode_listener = DMThread(self)
#         self.feedback_mode_listener = FMThread(self)
#         self.check_listener = CheckListener(self)
#         self.init_message_area()
#         self.init_function_area()
#         self.init_label_area()
#         self.init_image_area()

#         # short cut
#         QShortcut(QKeySequence(Qt.Key.Key_Right), self, self.check_listener.nxt)
#         QShortcut(QKeySequence(Qt.Key.Key_Left), self, self.check_listener.prev)
#         QShortcut(QKeySequence(Qt.CTRL + Qt.Key.Key_S), self, self.check_listener.update)
#         QShortcut(QKeySequence(Qt.CTRL + Qt.Key.Key_Q), self, self.check_listener.quit)
#         QShortcut(QKeySequence(Qt.CTRL + Qt.Key.Key_D), self, self.check_listener.delete)

    
#     def init_function_area(self):
#         # connect to server
#         self.ip_input.setText("127.0.0.1")
#         self.port_input.setText("9999")
#         self.connect_btn.clicked.connect(
#             lambda: connect_to_server(
#                 self, 
#                 self.ip_input.text().strip(), 
#                 int(self.port_input.text().strip()), 
#                 self.connect_btn, 
#                 self.select_learn, 
#                 self.message
#             )
#         )

#         # select mode
#         self.select_learn.setChecked(True)
#         self.select_detect.clicked.connect(
#             lambda: change_mode(self.listQRB, self.message, self.detect_mode_listener, self.feedback_mode_listener)
#         )
#         # self.select_feedback.clicked.connect(
#         #     lambda: change_mode(self.listQRB, self.message, self.detect_mode_listener, self.feedback_mode_listener)
#         # )
#         self.select_learn.clicked.connect(
#             lambda: change_mode(self.listQRB, self.message, self.detect_mode_listener, self.feedback_mode_listener)
#         )

#         # save button
#         self.save_button.pressed.connect(lambda:get_pressed(self.save_button))
#         self.save_button.released.connect(lambda: get_released(self.save_button))
#         self.save_button.clicked.connect(lambda: isSave(self.message, True))
#         self.unsave_button.pressed.connect(lambda:get_pressed(self.unsave_button))
#         self.unsave_button.released.connect(lambda: get_released(self.unsave_button))
#         self.unsave_button.clicked.connect(lambda: isSave(self.message, False))
#         # self.feedback_button.pressed.connect(lambda:get_pressed(self.feedback_button))
#         # self.feedback_button.released.connect(lambda:get_released(self.feedback_button))
#         # self.feedback_button.clicked.connect(lambda: send_feedback(self.message, self.listQRB, self))

#         # check button
#         self.check_btn.clicked.connect(lambda: check_data(self, self.check_btn, self.message, self.check_listener))

#     def init_image_area(self):
#         self.image_area.setCanvas(QPixmap("detect.png"))

#     def init_label_area(self):
#         # select
#         self.label_area.itemClicked.connect(self.image_area.canvas.selected_area_by_label)
#         # modify
#         self.label_area.itemDoubleClicked.connect(self.image_area.canvas.modify_label)
#         # clear all button
#         def clear_all():
#             self.message.set_labels([])
#             self.message.set_bboxes([])
#             self.image_area.canvas.pos_select_index = -1
#             self.label_area.update()
#             self.image_area.canvas.update_list_label()
#         self.label_button.clicked.connect(clear_all)

#     def init_message_area(self):
#         def update_message(message):
#             self.message_area.appendPlainText(message)
#         self.message_area.setMaximumBlockCount(1000)
#         self.message_area.setReadOnly(True)
#         self.message_area_signal.signal.connect(update_message)
from PyQt5.Qt import *
from PyQt5 import QtGui

from .const import *
from .common import *
from .listener import *
from .message import Message

class MainWindow(QWidget):

    def __init__(self, db):
        super(MainWindow, self).__init__()
        self.db = db
        self.setObjectName("MainWindow")
        self.message = Message()
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)

        # ----- message area -----
        self.message_area_title = QLabel("Message Area")
        self.message_area = QPlainTextEdit(self)
        self.message_area.setStyleSheet(MESSAGE_AREA_STYLE)
        self.message_area.setReadOnly(True)
        self.message_area.setMaximumBlockCount(1000)
        self.message_area_layout = QVBoxLayout()
        self.message_area_layout.addWidget(self.message_area_title)
        self.message_area_layout.addWidget(self.message_area)
        self.message_area_window = QWidget()
        self.message_area_window.setLayout(self.message_area_layout)
        self.message_area_window.setStyleSheet(ADD_MARGIN)
        self.message_area_signal = MessageAreaSignal()

        # ----- function area -----
        self.ip_label = QLabel("I   P:", self)
        self.ip_input = QLineEdit(self)
        self.ip_input.setStyleSheet(BORDER)
        self.port_label = QLabel("Port:", self)
        self.port_input = QLineEdit(self)
        self.port_input.setStyleSheet(BORDER)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet(BUTTON)

        self.ip_layout = QHBoxLayout()
        self.ip_layout.addWidget(self.ip_label)
        self.ip_layout.addWidget(self.ip_input)
        self.port_layout = QHBoxLayout()
        self.port_layout.addWidget(self.port_label)
        self.port_layout.addWidget(self.port_input)

        self.connect_layout = QVBoxLayout()
        self.connect_layout.addLayout(self.ip_layout)
        self.connect_layout.addLayout(self.port_layout)
        self.connect_layout.addWidget(self.connect_btn)

        self.connect_window = QWidget()
        self.connect_window.setLayout(self.connect_layout)

        # mode select
        self.listQRB = []
        self.select_detect = QRadioButton("DM")
        self.select_detect.setStyleSheet(BORDER)
        self.select_learn = QRadioButton("LM")
        self.select_learn.setStyleSheet(BORDER)
        self.listQRB.extend([self.select_detect, self.select_learn])

        self.select_mode_layout = QHBoxLayout()
        self.select_mode_layout.addWidget(self.select_detect)
        self.select_mode_layout.addWidget(self.select_learn)
        self.select_mode_window = QWidget()
        self.select_mode_window.setLayout(self.select_mode_layout)

        # check button
        self.check_btn = QPushButton("Check")
        self.check_btn.setStyleSheet(BUTTON)
        self.check_btn_window = QWidget()
        self.check_btn_layout = QVBoxLayout()
        self.check_btn_layout.addWidget(self.check_btn)
        self.check_btn_window.setLayout(self.check_btn_layout)
        self.on_check = False

        # layout for function area
        self.function_area_layout = QVBoxLayout()
        self.function_area_layout.addWidget(self.connect_window)
        self.function_area_layout.addWidget(self.select_mode_window)
        self.function_area_layout.addWidget(self.check_btn_window)

        self.function_area_window = QWidget()
        self.function_area_window.setLayout(self.function_area_layout)
        self.function_area_window.setStyleSheet(FUNCTION_COLUMN_STYLE)

        self.function_area_title = QLabel("Function Area")
        self.function_area_layout_with_title = QVBoxLayout()
        self.function_area_layout_with_title.addWidget(self.function_area_title)
        self.function_area_layout_with_title.addWidget(self.function_area_window)

        self.function_area_window_with_title = QWidget()
        self.function_area_window_with_title.setLayout(self.function_area_layout_with_title)

        # main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.function_area_window_with_title, 2)
        self.main_layout.addWidget(self.message_area_window, 8)
        self.setLayout(self.main_layout)

        self.setStyleSheet(MAIN_WINDOW_STYLE)

        self.detect_mode_listener = DMThread(self)
        self.feedback_mode_listener = FMThread(self)
        self.check_listener = CheckListener(self)

        self.init_function_area()
        self.init_message_area()

    def init_function_area(self):
        self.ip_input.setText("127.0.0.1")
        self.port_input.setText("9999")

        self.connect_btn.clicked.connect(
            lambda: connect_to_server(
                self, 
                self.ip_input.text().strip(), 
                int(self.port_input.text().strip()), 
                self.connect_btn, 
                self.select_learn, 
                self.message
            )
        )

        self.select_learn.setChecked(True)
        self.select_detect.clicked.connect(
            lambda: change_mode(self.listQRB, self.message, self.detect_mode_listener, self.feedback_mode_listener)
        )
        self.select_learn.clicked.connect(
            lambda: change_mode(self.listQRB, self.message, self.detect_mode_listener, self.feedback_mode_listener)
        )

        self.check_btn.clicked.connect(lambda: check_data(self, self.check_btn, self.message, self.check_listener))

    def init_message_area(self):
        def update_message(msg):
            self.message_area.appendPlainText(msg)
        self.message_area_signal.signal.connect(update_message)