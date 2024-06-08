import os
import os.path as osp
import sys
import time

import numpy as np
from IoU_Tracker import tracker
from Processing import postprocess

# Define the root directory for raw data
raw_data_root = '/media/cycyang/sda1/EE443_final/data'

# Image dimensions
W, H = 1920, 1080

# List of cameras to test on
data_list = {
    'test': ['camera_0008', 'camera_0019', 'camera_0028']
}

# Sampling rate and visualization flag
sample_rate = 1  # Set to 1 to test on all frames
vis_flag = True  # Set to True to save visualizations

# Define experiment paths for tracking results, detections, and embeddings
exp_path = '/media/cycyang/sda1/EE443_final/runs/tracking/inference'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

det_path = '/media/cycyang/sda1/EE443_final/runs/detect/inference/txt'
emb_path = '/media/cycyang/sda1/EE443_final/runs/reid/inference'

def estimate_number_of_people(detections):
    """
    Estimate the number of people in the scene based on detections.

    Parameters:
    -----------
    detections : numpy.ndarray
        Array of detections where each row is (camera_id, -1, frame_id, x, y, w, h, confidence).

    Returns:
    --------
    max : int
        Maximum number of people detected in any single frame.
    """
    people = []
    for frame_id in range(0, 3600):
        inds = detections[:, 2] == frame_id
        cur_frame_detection = detections[inds]
        size = len(cur_frame_detection)
        if size > 0:
            people.append(size)
  
    max_people = np.max(np.array(people))

    return max_people

def filter_detections(txt_data, npy_data, threshold, ratio_threshold):
    """
    Filter detections based on confidence score and width-to-height ratio.

    Parameters:
    -----------
    txt_data : numpy.ndarray
        Array of detections.
    npy_data : numpy.ndarray
        Array of embeddings.
    threshold : float
        Confidence score threshold.
    ratio_threshold : float
        Width-to-height ratio threshold.

    Returns:
    --------
    filtered_txt_data : numpy.ndarray
        Filtered detections.
    filtered_npy_data : numpy.ndarray
        Filtered embeddings.
    """
    # Extract confidence scores from the txt data
    confidence_scores = txt_data[:, 7]

    # Calculate width-to-height ratios
    ratios = txt_data[:, 6] / txt_data[:, 5]

    # Filter based on confidence score and width-to-height ratio
    filtered_indices = np.where((confidence_scores >= threshold) & (ratios <= ratio_threshold))[0]

    # Filter txt_data and npy_data based on the indices
    filtered_txt_data = txt_data[filtered_indices]
    filtered_npy_data = npy_data[filtered_indices]

    return filtered_txt_data, filtered_npy_data

for split in ['test']:
    for folder in data_list[split]:
        # Define paths for detections, embeddings, and tracking results
        det_txt_path = os.path.join(det_path, f'{folder}.txt')
        emb_npy_path = os.path.join(emb_path, f'{folder}.npy')
        tracking_txt_path = os.path.join(exp_path, f'{folder}.txt')

        # Load detections and embeddings
        detection = np.loadtxt(det_txt_path, delimiter=',', dtype=None)
        embedding = np.load(emb_npy_path, allow_pickle=True)

        print(f"Getting bounding boxes from {det_txt_path} (number of detections: {len(detection)})")
        print(f"Getting features from {emb_npy_path} (number of embeddings: {len(embedding)})")

        # Extract camera ID from folder name
        camera_id = int(folder.split('_')[-1])
        print(f"Tracking on camera {camera_id}")
        
        # Define filtering thresholds
        CONFIDENCE_THRESHOLD = 0.3
        RATIO_THRESHOLD = 2.5

        # Filter detections and embeddings based on thresholds
        detection, embedding = filter_detections(detection, embedding, CONFIDENCE_THRESHOLD, RATIO_THRESHOLD)

        # Estimate the number of people in the scene
        number_of_people = estimate_number_of_people(detection)
        postprocessing = postprocess(number_of_people, cluster_method='kmeans')

        # Run the IoU tracker
        mot = tracker()
        tracklets = mot.run(detection, embedding)

        # Extract features from tracklets for clustering
        features = np.array([trk.final_features for trk in tracklets])

        # Run the post-processing step to merge tracklets
        labels = postprocessing.run(features)  # The label represents the final tracking ID, starting from 0

        tracking_result = []

        print('Writing Result ... ')

        # Prepare tracking results for saving
        for i, trk in enumerate(tracklets):
            final_tracking_id = labels[i] + 1  # Make it start from 1
            for idx in range(len(trk.boxes)):
                frame = trk.times[idx]
                x, y, w, h = trk.boxes[idx]
                result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera_id, final_tracking_id, frame, x - w / 2, y - h / 2, w, h)
                tracking_result.append(result)
        
        print('Save tracking results at {}'.format(tracking_txt_path))

        # Save tracking results to a file
        with open(tracking_txt_path, 'w') as f:
            f.writelines(tracking_result)

# The below section is commented out but provides an example of how to run the script
# if __name__ == "__main__":
#     # Example setup for validation or test set
#     camera = 75  # Test set
#     number_of_people = 5
#     result_path = 'baseline_result.txt'
#
#     # Load detection and embedding data
#     detection = np.loadtxt('../detection.txt', delimiter=',', dtype=None)
#     embedding = np.load('../embedding.npy', allow_pickle=True)
#     inds = detection[:, 0] == camera
#     test_detection = detection[inds]
#     test_embedding = embedding[inds]
#     sort_inds = test_detection[:, 1].argsort()
#     test_detection = test_detection[sort_inds]
#     test_embedding = test_embedding[sort_inds]
#
#     # Initialize tracker and post-processing
#     mot = tracker()
#     postprocessing = postprocess(number_of_people, 'kmeans')
#
#     # Run the IoU tracking
#     tracklets = mot.run(test_detection, test_embedding)
#
#     # Extract features for clustering
#     features = np.array([trk.final_features for trk in tracklets])
#
#     # Run the post-processing to merge tracklets
#     labels = postprocessing.run(features)  # Labels start from 0, will be adjusted to start from 1
#
#     tracking_result = []
#     print('Writing Result ... ')
#
#     # Prepare and save tracking results
#     for i, trk in enumerate(tracklets):
#         final_tracking_id = labels[i] + 1  # Make it start from 1
#         for idx in range(len(trk.boxes)):
#             frame = trk.times[idx]
#             x1, y1, x2, y2 = trk.boxes[idx]
#             x, y, w, h = x1, y1, x2 - x1, y2 - y1
#             result = '{},{},{},{},{},{},{},-1,-1 \n'.format(camera, final_tracking_id, frame, x, y, w, h)
#             tracking_result.append(result)
#     print('Save tracking results at {}'.format(result_path))
#     with open(result_path, 'w') as f:
#         f.writelines(tracking_result)
