import numpy as np
from scipy.optimize import linear_sum_assignment

# Calculate the Intersection over Union (IoU) of two bounding boxes
def calculate_iou(bbox1, bbox2):
    """
    Calculate the IoU of two bounding boxes.

    Parameters:
    -----------
    bbox1 : tuple
        The first bounding box (x1, y1, width, height).
    bbox2 : tuple
        The second bounding box (x1, y1, width, height).

    Returns:
    --------
    iou : float
        The Intersection over Union (IoU) of the two bounding boxes.
    """
    # Extract the coordinates and dimensions of the bounding boxes
    x1_1, y1_1, w1, h1 = bbox1
    x1_2, y1_2, w2, h2 = bbox2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    # Calculate the coordinates of the intersection rectangle
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # If there is no intersection, return 0.0
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the area of the bounding boxes and their intersection
    area_bbox1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
    area_bbox2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # Calculate the IoU
    iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)

    return iou

# Base class for a tracklet
class tracklet:
    """
    A class to represent a tracklet, which is a sequence of bounding boxes and features over time.

    Attributes:
    -----------
    ID : int
        The tracking ID of the tracklet.
    boxes : list
        A list of bounding boxes associated with the tracklet.
    features : list
        A list of features associated with the tracklet.
    times : list
        A list of timestamps associated with the tracklet.
    cur_box : tuple
        The current bounding box of the tracklet.
    cur_feature : None
        The current feature of the tracklet (not used in this implementation).
    alive : bool
        A flag indicating whether the tracklet is still active.
    final_features : None or numpy.ndarray
        The average of the features associated with the tracklet.
    """
    
    def __init__(self, tracking_ID, box, feature, time):
        self.ID = tracking_ID
        self.boxes = [box]
        self.features = [feature]
        self.times = [time]

        self.cur_box = box
        self.cur_feature = None
        self.alive = True

        self.final_features = None
    
    def update(self, box, feature, time):
        """
        Update the tracklet with a new bounding box, feature, and time.

        Parameters:
        -----------
        box : tuple
            The new bounding box.
        feature : numpy.ndarray
            The new feature.
        time : int
            The new timestamp.
        """
        self.cur_box = box
        self.boxes.append(box)
        self.cur_feature = None  # You might need to do the update if you also want to use features for tracking
        self.features.append(feature)
        self.times.append(time)
    
    def close(self):
        """Mark the tracklet as no longer active."""
        self.alive = False
    
    def get_avg_features(self):
        """Calculate the average of the features associated with the tracklet."""
        self.final_features = sum(self.features) / len(self.features)

# Class for multi-object tracker
class tracker:
    """
    A class to represent a multi-object tracker.

    Attributes:
    -----------
    all_tracklets : list
        A list of all tracklets.
    cur_tracklets : list
        A list of currently active tracklets.
    
    Methods:
    --------
    run(detections, features=None):
        Run the tracker on the given detections and features.
    """
    
    def __init__(self):
        self.all_tracklets = []
        self.cur_tracklets = []

    def run(self, detections, features=None):
        """
        Run the tracker on the given detections and features.

        Parameters:
        -----------
        detections : numpy.ndarray
            An array of detections where each row is (frame_id, detection_id, frame_number, x, y, width, height).
        features : numpy.ndarray or None, optional
            An array of features associated with the detections. Default is None.

        Returns:
        --------
        final_tracklets : list
            A list of final tracklets after processing all frames.
        """
        
        for frame_id in range(0, 3600):

            if frame_id % 100 == 0:
                print('Tracking | cur_frame {} | total frame 3600'.format(frame_id))

            # Get detections for the current frame
            inds = detections[:, 2] == frame_id
            cur_frame_detection = detections[inds]
            if features is not None:
                cur_frame_features = features[inds]
            
            # No tracklets in the first frame
            if len(self.cur_tracklets) == 0:
                for idx in range(len(cur_frame_detection)):
                    new_tracklet = tracklet(len(self.all_tracklets) + 1, cur_frame_detection[idx][3:7], cur_frame_features[idx], frame_id)
                    self.cur_tracklets.append(new_tracklet)
                    self.all_tracklets.append(new_tracklet)
            
            else:
                # Create a cost matrix based on IoU
                cost_matrix = np.zeros((len(self.cur_tracklets), len(cur_frame_detection)))

                for i in range(len(self.cur_tracklets)):
                    for j in range(len(cur_frame_detection)):
                        cost_matrix[i][j] = 1 - calculate_iou(self.cur_tracklets[i].cur_box, cur_frame_detection[j][3:7])
                
                # Solve the assignment problem
                row_inds, col_inds = linear_sum_assignment(cost_matrix)

                matches = min(len(row_inds), len(col_inds))

                for idx in range(matches):
                    row, col = row_inds[idx], col_inds[idx]
                    if cost_matrix[row, col] == 1:
                        # Close the current tracklet and start a new one
                        self.cur_tracklets[row].close()
                        new_tracklet = tracklet(len(self.all_tracklets) + 1, cur_frame_detection[col][3:7], cur_frame_features[col], frame_id)
                        self.cur_tracklets.append(new_tracklet)
                        self.all_tracklets.append(new_tracklet)
                    else:
                        # Update the matched tracklet
                        self.cur_tracklets[row].update(cur_frame_detection[col][3:7], cur_frame_features[col], frame_id)

                # Initiate unmatched detections as new tracklets
                for idx, det in enumerate(cur_frame_detection):
                    if idx not in col_inds:  # If it is not matched in the above Hungarian algorithm stage
                        new_tracklet = tracklet(len(self.all_tracklets) + 1, det[3:7], cur_frame_features[idx], frame_id)
                        self.cur_tracklets.append(new_tracklet)
                        self.all_tracklets.append(new_tracklet)
            
            # Remove dead tracklets
            self.cur_tracklets = [trk for trk in self.cur_tracklets if trk.alive]            

        final_tracklets = self.all_tracklets

        # Calculate the average final features for all the tracklets
        for trk_id in range(len(final_tracklets)):
            final_tracklets[trk_id].get_avg_features()

        return final_tracklets
