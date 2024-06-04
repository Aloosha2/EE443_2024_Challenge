# EE 443 2024 Challenge: Single Camera Multi-Object Tracking

# Team name: Amnesty

Here is how the repo is organized per folder:

detection: Contains the models as well as the provided starter code. The trained models are saved here, as well as the inference results for test and validation sets. 

evaluation: Contains the script used to evaluate the accuracy. We used camera 5 for evaluation. Here, "camera_0005_gt.txt" is the ground truth, and "camera_0005.txt" is the tracking result.

reid: Includes the feature extraction code provided by the TA as well as the model weights. 

tracking: Contains the tracking code provided by the TA.

Additional created files:

concat.ipynb: Performs the concatenation of two numpy files taken from the feature extraction step as an attempt to improve performance.

filter.py: Filters out the low-confidence values from the model results as an attempt to improve performance. 