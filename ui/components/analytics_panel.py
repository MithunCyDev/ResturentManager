from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel

class AnalyticsPanel(QGroupBox):
    def __init__(self):
        super().__init__("Real-time Analytics")
        self.init_ui()
        
    def init_ui(self):
        self.layout = QVBoxLayout()
        self.metrics = {}
        
        metrics = [
            ("Current Customers", "#4CAF50"),
            ("Total Entries", "#2196F3"),
            ("Total Exits", "#F44336"),
            ("Coffee Orders", "#6D4C41"),
            ("Meal Orders", "#FF9800")
        ]
        
        for title, color in metrics:
            self.add_metric(title, color)
            
        self.setLayout(self.layout)
        
    def add_metric(self, title, color):
        widget = QWidget()
        # Metric implementation
        self.metrics[title] = (title, color, QLabel("0"))
        self.layout.addWidget(widget)
        
    def update_metrics(self, data):
        # Update logic
        pass