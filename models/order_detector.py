from ultralytics import YOLO

import os
from pathlib import Path

class OrderDetector:
    def __init__(self, model_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file missing: {model_path}\n"
                                    "1. Download pre-trained model:\n"
                                    "   from ultralytics import YOLO; YOLO('yolov8n.pt')\n"
                                    "2. Or train your own model")
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        results = self.model(frame)
        return self.process_results(results)

    def process_results(self, results):
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': result.names[box.cls[0].item()],
                    'confidence': box.conf[0].item(),
                    'bbox': box.xyxy[0].tolist()
                })
        return detections