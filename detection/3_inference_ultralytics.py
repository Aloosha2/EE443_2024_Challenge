import os
from ultralytics import YOLO

# Path to the root directory containing raw data
raw_data_root = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/data/data'

# Image dimensions
W, H = 1920, 1080

# List of camera folders for different splits
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028'],
    'val': ['camera_0005', 'camera_0017', 'camera_0025'],
}

# Sampling rate for frames
sample_rate = 1

# Flag to save visualizations
vis_flag = True

# Path for saving inference results
exp_path = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/detection/runs/detect/inference'

# Path to the trained model weights
model_path = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/detection/runs/detect/train2/weights/best.pt'

# Initialize YOLO detector with the trained model
det_model = YOLO(model_path)

for split in ['test']:
    for folder in data_list[split]:

        # Path to the folder containing images for the current camera
        camera_img_folder = os.path.join(raw_data_root, split, folder)
        camera_img_list = os.listdir(camera_img_folder)
        camera_img_list.sort()

        # Extract camera ID from folder name
        camera_id = int(folder.split('_')[-1])
        lines_to_write = []

        # Create folder for saving visualizations
        if vis_flag:
            if not os.path.exists(os.path.join(exp_path, 'vis', folder)):
                os.makedirs(os.path.join(exp_path, 'vis', folder))

        # Process each image in the camera folder
        for img_name in camera_img_list:
            frame_id = int(img_name.split('.')[0])

            # Perform object detection on the image
            results = det_model(camera_img_folder + '/' + img_name)
            boxes = results[0].boxes.xywh.cpu().numpy().tolist()
            confs = results[0].boxes.conf.cpu().numpy().tolist()

            # Save visualizations if enabled
            if vis_flag:
                save_vis_img_path = os.path.join(exp_path, 'vis', folder, img_name)   
                results[0].save(filename=save_vis_img_path)
            
            # Format detection results and append to lines_to_write
            for box, conf in zip(boxes, confs):
                x, y, w, h = box
                lines_to_write.append(f'{camera_id}, -1, {frame_id}, {x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}, {conf:.3f},-1')

        # Write the detection results to a file
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        with open(os.path.join(exp_path, 'txt', f'{folder}.txt'), 'w') as f:
            f.write('\n'.join(lines_to_write))
