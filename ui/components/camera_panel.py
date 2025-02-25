import cv2
cv2.dnn_DictValue = None  # Workaround for typing issue

# Rest of your imports
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QTimer, Qt

class CameraPanel(QFrame):
    def __init__(self, camera_id, face_detector, order_detector):
        super().__init__()
        self.camera_id = camera_id
        self.face_detector = face_detector
        self.order_detector = order_detector
        self.init_ui()
        self.init_camera()
        
    def init_ui(self):
        self.setFixedSize(400, 300)
        self.layout = QVBoxLayout()
        
        # Feed Display
        self.feed_label = QLabel()
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.feed_label)
        
        # Controls
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        
        self.setLayout(self.layout)
        
    def init_camera(self):
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        self.timer.start(config.REFRESH_RATE)
        
    def stop_camera(self):
        self.timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)
            self.display_frame(frame)
            
    def process_frame(self, frame):
        # Face detection
        detections = self.face_detector.detect(frame)
        # Order detection
        orders = self.order_detector.detect(frame)
        # Add processing logic
        return frame
    
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.feed_label.setPixmap(QPixmap.fromImage(q_img))