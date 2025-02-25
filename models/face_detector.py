import cv2

class FaceDetector:
    def __init__(self, proto_path, model_path):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        
    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        return self.net.forward()