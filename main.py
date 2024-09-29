from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.yaml')
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data='config.yaml', epochs=100)

