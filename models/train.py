from ultralytics import YOLO

# Train a custom model
model = YOLO('yolov8n.yaml')  # Build new model
# model = YOLO('yolov8n.pt')  # Transfer learning

results = model.train(
    data='your_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)