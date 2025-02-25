from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout
from ui.components.camera_panel import CameraPanel
from ui.components.analytics_panel import AnalyticsPanel
from models.face_detector import FaceDetector
from models.order_detector import OrderDetector
from config.settings import config

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.face_detector = FaceDetector("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        self.order_detector = OrderDetector("best.pt")
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Smart Restaurant Manager")
        self.setGeometry(100, 100, 1280, 720)
        
        main_layout = QHBoxLayout()
        
        # Camera Grid
        self.camera_grid = QGridLayout()
        self.cameras = []
        for i in range(config.MAX_CAMERAS):
            camera = CameraPanel(i, self.face_detector, self.order_detector)
            self.cameras.append(camera)
            self.camera_grid.addWidget(camera, i//2, i%2)
            
        # Analytics Panel
        self.analytics = AnalyticsPanel()
        
        main_layout.addLayout(self.camera_grid, 1)
        main_layout.addWidget(self.analytics)
        
        self.setLayout(main_layout)