from ultralytics import YOLO


# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='/Users/alicjamisiuda/UW/ee443/final_project/EE443_2024_Challenge/detection/ee443.yaml', epochs=100, imgsz=640)