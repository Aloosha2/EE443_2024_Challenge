import numpy as np

# Load the numpy arrays from the specified files
a = np.load('camera_0005_os_ain.npy', allow_pickle=True)
b = np.load('camera_0005_os.npy', allow_pickle=True)

# Initialize an empty list to store the concatenated results
result = []

# Iterate over the loaded arrays and concatenate corresponding elements
for one, two in zip(a, b):
    conc = np.concatenate((one, two))
    result.append(conc)

# Convert the list of concatenated results to a numpy array
result = np.array(result)

# Save the concatenated result to a new numpy file
np.save('camera_0005.npy', result)
