import sys

from PyQt5.Qt import *
from PyQt5 import QtGui
from .dialog import LabelDialog

class Position:

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class MyLabel(QLabel):

    def __init__(self, parent, labeled_list, list_item=["defect"], origin=True, message=None):
        super(MyLabel, self).__init__(parent) 
        self.message = message

        self.pos_press = Position() # mouse press pos
        self.pos_release = Position() # mouse release pos
        self.pos_select = Position() # right mouse press pos
        self.pos_select_index = -1 #  right mouse select pos index
        
        self.PRESSED = False
        self.list_item = list_item
        self.label_dialog = LabelDialog(parent=self, list_item=list_item)
        self.label_dialog.setStyleSheet("background-color: #000000;")
        
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.labeled_list = labeled_list
        self.class_name = list_item

        # delete selected label.
        if origin:
            self.delete_selected_area_action = Canvas.new_action(self, "delete selected area", self, Qt.Key_Delete, self.delete_selected_area)
        self.origin = origin
        
    def update_list_label(self):
        self.pos_select_index = -1
        self.labeled_list.clear()
        for label in self.message.get_labels():     
                self.labeled_list.addItem(label)
        self.update()

    def delete_selected_area(self):
        if not self.origin:
            return 
        if self.pos_select_index < 0:
            pass
        else:
            self.message.get_labels().pop(self.pos_select_index)
            self.message.get_bboxes().pop(self.pos_select_index)
            self.pos_select_index = -1
            self.labeled_list.clear()
            for label in self.message.get_labels():     
                self.labeled_list.addItem(label)
            self.update()


    def mouseMoveEvent(self, event):
        if not self.origin:
            return 
        if self.PRESSED == True:
            self.pos_release.x = event.x()/self.scale_x
            self.pos_release.y = event.y()/self.scale_y
            self.update()    
            print(f"{self.pos_release.x}  {self.pos_release.y}")
    
    def mousePressEvent(self, event):
        # origin area could be drawn but detection area cannot
        if not self.origin:
            return 
        # left button pressed means draw rectangle, and right means slect rectangle.
        if not self.PRESSED and event.button() == Qt.LeftButton:
            self.PRESSED = True
            self.pos_select_index = -1
            self.pos_press.x = event.x()/self.scale_x
            self.pos_press.y = event.y()/self.scale_y
            self.pos_release.x = self.pos_press.x
            self.pos_release.y = self.pos_press.y
        else:
            self.pos_select.x = event.x()/self.scale_x
            self.pos_select.y = event.y()/self.scale_y
            for index, bbox in enumerate(self.message.get_bboxes()):
                if self.pos_select.x > bbox[0] and self.pos_select.x < bbox[0] + bbox[2] and self.pos_select.y > bbox[1] and self.pos_select.y < bbox[1] + bbox[3]:
                    # a box is selected, record the index of selected box.
                    self.pos_select_index = index
                    break
                else:
                    self.pos_select_index = -1
            if self.pos_select_index >= 0:
                item = self.labeled_list.item(self.pos_select_index)
                item.setSelected(True)
            self.update()
    
    def mouseReleaseEvent(self, event):
        # origin area could be drawn but detection area cannot
        if not self.origin:
            return 
        # left button pressed means draw rectangle, and right means none.
        if self.PRESSED and event.button() == Qt.LeftButton:
            self.PRESSED = False
            self.pos_release.x = event.x()/self.scale_x
            self.pos_release.y = event.y()/self.scale_y
            x = min(self.pos_press.x, self.pos_release.x)
            y = min(self.pos_press.y, self.pos_release.y)
            w = abs(self.pos_press.x - self.pos_release.x)
            h = abs(self.pos_press.y - self.pos_release.y)
            label = self.label_dialog.pop_up()
            if label is not None:
                self.message.get_bboxes().append([x, y, w, h])
                self.message.get_labels().append(label)
                self.pos_press.y = self.pos_release.y
                if label not in self.list_item:
                    self.label_dialog.list_widget.addItem(label)
                    self.class_name.append(label)
                self.list_item.append(label)
                self.labeled_list.clear()
                for label in self.message.get_labels():
                    self.labeled_list.addItem(label)
            else:
                self.pos_press.y = self.pos_release.y
    
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        for index, (bbox, label) in enumerate(zip(self.message.get_bboxes(), self.message.get_labels())):
            if self.pos_select_index == index:
                continue
            rect = QRect(int(bbox[0]*self.scale_x), 
                         int(bbox[1]*self.scale_y), 
                         int(bbox[2]*self.scale_x), 
                         int(bbox[3]*self.scale_y))
            if label.endswith("background"):
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            else:
                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.drawRect(rect)
            painter.drawText(int(bbox[0]*self.scale_x), int(bbox[1]*self.scale_y), label)
        # paint selected box
        if self.pos_select_index >= 0:
            bbox = self.message.get_bboxes()[self.pos_select_index]
            label = self.message.get_labels()[self.pos_select_index]
            rect = QRect(int(bbox[0]*self.scale_x), 
                         int(bbox[1]*self.scale_y), 
                         int(bbox[2]*self.scale_x), 
                         int(bbox[3]*self.scale_y))
            if label.endswith("background"):
                painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
            else:
                painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
            painter.setBrush(QBrush(Qt.red, Qt.CrossPattern))
            painter.drawRect(rect)
            painter.drawText(int(bbox[0]*self.scale_x), int(bbox[1]*self.scale_y), label)
        # paint current area
        x = min(self.pos_press.x, self.pos_release.x)
        y = min(self.pos_press.y, self.pos_release.y)
        w = abs(self.pos_press.x - self.pos_release.x)
        h = abs(self.pos_press.y - self.pos_release.y)
        if w == 0 or h == 0:
            return 
        painter.setPen(QPen(Qt.green, 2, Qt.SolidLine))
        rect = QRect(int(x*self.scale_x), int(y*self.scale_y), int(w*self.scale_x), int(h*self.scale_y))
        painter.drawRect(rect)
    
    def selected_area_by_label(self):
        """Select bounding box and label (single click).
        
        """
        self.pos_select_index = self.labeled_list.currentRow()
        self.update()
    
    def modify_label(self):
        """modify  label (double click).
        
        """
        while True:
            label = self.label_dialog.pop_up()
            if label is not None:
                break
            else:
                return

        self.message.get_labels()[self.labeled_list.currentRow()] = label
        if label not in self.list_item:
            self.label_dialog.list_widget.addItem(label)
            self.class_name.append(label)
        self.list_item.append(label)
        self.labeled_list.clear()
        for label in self.message.get_labels():
            self.labeled_list.addItem(label)
        self.update()



class Canvas(QWidget):
    """Component to edit image.

    Args:
        parent: parent widget.
        origin: True: image can be edited. False: image can be not edited. 
        message: communicate with server.
        list_item: datasets class name.
        labeled_list: current image label list, show in label area.
    
    """
    
    def __init__(self, parent, labeled_list, origin=False, message=None, list_item=None):
        super(Canvas, self).__init__(parent) 
        self.message = message
        
        if list_item is None:
            self.canvas = MyLabel(self, labeled_list, origin=origin, message=self.message)
        else:
            self.canvas = MyLabel(self, labeled_list, origin=origin, message=self.message, list_item=list_item)
        self.canvas.setScaledContents(True)
        self.canvas.setCursor(Qt.CrossCursor)

        # add the canvas to the scroll
        self.scroll = QScrollArea(self) 
        self.scroll.setWidget(self.canvas) 

        self.canvas_layout = QHBoxLayout()
        self.canvas_layout.addWidget(self.scroll)
        self.setLayout(self.canvas_layout)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)
        
    
    def setCanvas(self, qpixmap):

        qpixmap = QtGui.QPixmap.copy(qpixmap)
        # init canvas
        self.canvas.scale_x = 1.0
        self.canvas.scale_y = 1.0
        self.canvas.pos_select_index = -1
        # set image
        self.canvas.setPixmap(qpixmap)
        self.img_width = qpixmap.width()
        self.img_height = qpixmap.height()
        self.canvas.scale_x = self.scroll.width()/self.img_width
        self.canvas.scale_y = self.scroll.height()/self.img_height
        
        self.canvas.resize(int(self.img_width*self.canvas.scale_x), int(self.img_height*self.canvas.scale_y))

    def resizeEvent(self, event):
        self.canvas.scale_x = self.scroll.width()/self.img_width
        self.canvas.scale_y = self.scroll.height()/self.img_height
        self.canvas.resize(int(self.img_width*self.canvas.scale_x), int(self.img_height*self.canvas.scale_y))


    @staticmethod
    def new_action(component, text, parent, shortcut, slot):
        action = QAction(text, parent)
        action.setShortcut(shortcut)
        action.triggered.connect(slot)
        component.addAction(action)
    



if __name__ == "__main__":
    app = QApplication(sys.argv)
    canvas = Canvas()
    canvas.show()
    sys.exit(app.exec_())
