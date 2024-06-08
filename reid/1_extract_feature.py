import os
import os.path as osp
import sys
import time

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from argparse import ArgumentParser

import torchreid
from torchreid.reid.utils import FeatureExtractor

# Define the root directory for the raw data
raw_data_root = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/data/data'

# Define the width and height of the images
W, H = 1920, 1080

# Define the list of data for testing and validation
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028'],
    'val':  ['camera_0005', 'camera_0017', 'camera_0025']
}

# Define the sample rate for frames
sample_rate = 1  # We want to test on all frames

# Define paths for detection and experimental results
det_path = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/detection/runs/detect/inference/txt'
exp_path = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/runs/reid/inference'
reid_model_ckpt = 'C:/Users/konya/Desktop/UW/ee classes/ee443/EE443_2024_Challenge/reid/osnet_x1_0_imagenet.pth'

# Define the transformations to be applied to the validation images
val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize the ReID feature extractor with the specified model and parameters
reid_extractor = FeatureExtractor(
    model_name='osnet_x1_0',  # Confirmed to be the best model via trial and error
    model_path=reid_model_ckpt,
    image_size=[256, 128],
    device='cuda'
)

# Iterate over the splits (e.g., 'test')
for split in ['test']:
    # Iterate over each folder in the split
    for folder in data_list[split]:
        # Define the path to the detection file
        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        print(f"Extracting feature from {det_txt_path}")

        # Load the detection data from the file
        dets = np.genfromtxt(det_txt_path, dtype=str, delimiter=',')

        # Initialize the feature array with None values
        emb = np.array([None] * len(dets))

        # Iterate over the detection data
        for idx, (camera_id, _, frame_id, x, y, w, h, score, _) in enumerate(dets):
            # Convert bounding box coordinates to float
            x, y, w, h = map(float, [x, y, w, h])
            # Convert frame ID to string and remove leading spaces
            frame_id = str(int(frame_id))

            # Print progress every 1000 frames
            if idx % 1000 == 0:
                print(f'Processing frame {frame_id} | {idx}/{len(dets)}')

            # Define the path to the image file
            img_path = os.path.join(raw_data_root, split, folder, frame_id.zfill(5) + '.jpg')
            img = Image.open(img_path)

            # Crop the image according to the bounding box coordinates
            img_crop = img.crop((x-w/2, y-h/2, x+w/2, y+h/2))
            # Apply the transformations to the cropped image
            img_crop = val_transforms(img_crop.convert('RGB')).unsqueeze(0)
            # Extract features using the ReID extractor
            feature = reid_extractor(img_crop).cpu().detach().numpy()[0]

            # Normalize the extracted feature
            feature = feature / np.linalg.norm(feature)
            emb[idx] = feature

        # Define the path to save the embedding file
        emb_save_path = os.path.join(exp_path, f'{folder}.npy')
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        # Save the embedding to a file
        np.save(emb_save_path, emb)
