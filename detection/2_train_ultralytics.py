from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8x.yaml')  # Changed model to yolov8x
    model.to('cuda')
    print(torch.cuda.is_available())
    # Train the model
    results = model.train(data='C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/detection/ee443.yaml', epochs=100, batch=-1, imgsz=640)
