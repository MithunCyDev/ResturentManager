import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QGridLayout, QFrame, QGroupBox)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt, QMutex, QPoint
from datetime import datetime
import os

# Face detection model paths
FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
    raise FileNotFoundError("Missing face detection model files! Download them from OpenCV's GitHub repository.")

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

class Customer:
    def __init__(self, customer_id, entry_time):
        self.id = customer_id
        self.entry_time = entry_time
        self.exit_time = None
        self.orders = {'coffee': 0, 'meal': 0}
        self.last_seen = datetime.now()
        self.in_store = True
        self.position = None

class RestaurantTracker:
    def __init__(self):
        self.mutex = QMutex()
        self.customers = {}
        self.next_id = 1
        self.entry_count = 0
        self.exit_count = 0
        self.order_counts = {'coffee': 0, 'meal': 0}
        self.entrance_line = 0.2  # 20% from top
        self.exit_line = 0.8      # 80% from top

    def detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        return face_net.forward()

    def update_tracking(self, frame):
        self.mutex.lock()
        try:
            h, w = frame.shape[:2]
            detections = self.detect_faces(frame)
            current_ids = []

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    centroid = QPoint((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    customer_id = self.next_id
                    if customer_id not in self.customers:
                        self.customers[customer_id] = Customer(customer_id, datetime.now())
                        self.next_id += 1
                    
                    self.customers[customer_id].position = centroid
                    self.customers[customer_id].last_seen = datetime.now()
                    current_ids.append(customer_id)
            
            # Remove old entries
            to_remove = [cid for cid in self.customers if cid not in current_ids]
            for cid in to_remove:
                if self.customers[cid].in_store:
                    self.exit_count += 1
                    self.customers[cid].exit_time = datetime.now()
                    self.customers[cid].in_store = False
            
            self.entry_count = len(self.customers)
            self.check_orders(frame)
            
        finally:
            self.mutex.unlock()

    def check_orders(self, frame):
        for customer in self.customers.values():
            if customer.in_store:
                if np.random.rand() < 0.05:
                    customer.orders['coffee'] += 1
                    self.order_counts['coffee'] += 1
                if np.random.rand() < 0.03:
                    customer.orders['meal'] += 1
                    self.order_counts['meal'] += 1

class CameraSystem(QFrame):
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.tracker = RestaurantTracker()
        self.is_streaming = False
        self.setup_ui()
        self.check_camera_availability()

    def setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                border: 2px solid #444;
                border-radius: 8px;
                background-color: #1E1E1E;
                min-width: 400px;
                min-height: 300px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Camera feed display
        self.feed_label = QLabel("Camera Feed")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet("color: #AAA; font: 14px;")
        layout.addWidget(self.feed_label)
        
        # Status bar
        self.status_label = QLabel("Status: Unknown")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #AAA; font: 12px;")
        layout.addWidget(self.status_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        
        button_style = """
            QPushButton {
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
                font: bold 12px;
            }
            QPushButton:disabled { background-color: #505050; }
        """
        self.start_btn.setStyleSheet(button_style + "background-color: #107C10;")
        self.stop_btn.setStyleSheet(button_style + "background-color: #D83B01;")
        
        self.start_btn.clicked.connect(self.start_feed)
        self.stop_btn.clicked.connect(self.stop_feed)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
        
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_feed)

    def check_camera_availability(self):
        temp_cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            self.status_label.setText("Camera Ready")
            self.status_label.setStyleSheet("color: #4CAF50;")
            self.start_btn.setEnabled(True)
            temp_cap.release()
        else:
            self.status_label.setText("No Camera Detected")
            self.status_label.setStyleSheet("color: #F44336;")
            self.start_btn.setEnabled(False)

    def start_feed(self):
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if self.capture.isOpened():
            self.is_streaming = True
            self.timer.start(30)
            self.status_label.setText("Streaming")
            self.status_label.setStyleSheet("color: #2196F3;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.status_label.setText("Connection Failed")
            self.status_label.setStyleSheet("color: #F44336;")

    def stop_feed(self):
        self.is_streaming = False
        self.timer.stop()
        if self.capture and self.capture.isOpened():
            self.capture.release()
        self.feed_label.clear()
        self.status_label.setText("Camera Ready")
        self.status_label.setStyleSheet("color: #4CAF50;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_feed(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.tracker.update_tracking(frame)
            self.draw_analytics(frame)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.feed_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.feed_label.size(), Qt.KeepAspectRatio))

    def draw_analytics(self, frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (0, int(h * self.tracker.entrance_line)),
                 (w, int(h * self.tracker.entrance_line)), (0, 255, 0), 2)
        cv2.line(frame, (0, int(h * self.tracker.exit_line)),
                 (w, int(h * self.tracker.exit_line)), (0, 0, 255), 2)
        
        for customer in self.tracker.customers.values():
            if customer.position and customer.in_store:
                cv2.putText(frame, f"ID: {customer.id}", 
                            (customer.position.x(), customer.position.y()),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

class RestaurantAnalytics(QGroupBox):
    def __init__(self):
        super().__init__("Restaurant Analytics")
        self.setFixedWidth(350)
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #444;
                border-radius: 8px;
                margin-top: 10px;
                color: #FFF;
                font: bold 14px;
                padding: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                color: #AAAAAA;
            }
        """)
        
        layout = QVBoxLayout()
        
        metrics = [
            ("Current Customers", "#4CAF50", "0"),
            ("Total Entries", "#2196F3", "0"),
            ("Total Exits", "#F44336", "0"),
            ("Coffee Orders", "#6D4C41", "0"),
            ("Meal Orders", "#FF9800", "0")
        ]
        
        self.value_labels = {}
        for title, color, value in metrics:
            widget = QWidget()
            hbox = QHBoxLayout()
            
            dot = QLabel("•")
            dot.setStyleSheet(f"color: {color}; font: bold 24px;")
            hbox.addWidget(dot)
            
            vbox = QVBoxLayout()
            title_label = QLabel(title)
            title_label.setStyleSheet("color: #AAA; font: 12px;")
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {color}; font: bold 18px;")
            
            vbox.addWidget(title_label)
            vbox.addWidget(value_label)
            hbox.addLayout(vbox)
            
            widget.setLayout(hbox)
            layout.addWidget(widget)
            self.value_labels[title] = value_label
        
        layout.addStretch()
        self.setLayout(layout)

    def update_metrics(self, data):
        self.value_labels["Current Customers"].setText(str(data['current']))
        self.value_labels["Total Entries"].setText(str(data['entries']))
        self.value_labels["Total Exits"].setText(str(data['exits']))
        self.value_labels["Coffee Orders"].setText(str(data['coffee']))
        self.value_labels["Meal Orders"].setText(str(data['meals']))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Restaurant Manager")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #252526;")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Camera Grid (2x2)
        camera_grid = QGridLayout()
        camera_grid.setSpacing(15)
        self.cameras = []
        
        for i in range(4):
            camera = CameraSystem(i)
            self.cameras.append(camera)
            camera_grid.addWidget(camera, i//2, i%2)
        
        # Analytics Panel
        self.analytics = RestaurantAnalytics()
        
        # Control Panel
        control_panel = QFrame()
        control_layout = QHBoxLayout()
        self.stop_all_btn = QPushButton("Stop All Cameras")
        self.exit_btn = QPushButton("Exit System")
        
        # Button styling
        btn_style = """
            QPushButton {
                background-color: #0078D4;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font: bold 14px;
                min-width: 150px;
            }
            QPushButton:hover { background-color: #006CBB; }
        """
        self.stop_all_btn.setStyleSheet(btn_style.replace("#0078D4", "#D83B01"))
        self.exit_btn.setStyleSheet(btn_style)
        
        control_layout.addWidget(self.stop_all_btn)
        control_layout.addWidget(self.exit_btn)
        control_panel.setLayout(control_layout)
        
        # Main layout
        main_layout.addLayout(camera_grid, 1)
        main_layout.addWidget(self.analytics)
        
        # Add control panel to bottom
        main_layout.addWidget(control_panel)
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_analytics)
        self.update_timer.start(1000)
        
        # Connect signals
        self.stop_all_btn.clicked.connect(self.stop_all_cameras)
        self.exit_btn.clicked.connect(self.close)
        
        self.setLayout(main_layout)

    def update_analytics(self):
        total_data = {
            'current': 0,
            'entries': 0,
            'exits': 0,
            'coffee': 0,
            'meals': 0
        }
        
        for camera in self.cameras:
            if camera.is_streaming:
                total_data['current'] += camera.tracker.entry_count - camera.tracker.exit_count
                total_data['entries'] += camera.tracker.entry_count
                total_data['exits'] += camera.tracker.exit_count
                total_data['coffee'] += camera.tracker.order_counts['coffee']
                total_data['meals'] += camera.tracker.order_counts['meal']
        
        self.analytics.update_metrics(total_data)

    def stop_all_cameras(self):
        for camera in self.cameras:
            if camera.is_streaming:
                camera.stop_feed()

    def closeEvent(self, event):
        self.stop_all_cameras()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())