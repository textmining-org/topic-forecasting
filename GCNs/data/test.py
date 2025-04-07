import numpy as np

# Load the .npy file
file_path = "betweenness_centrality.inv_cooccurrence.node_targets.npy"
data = np.load(file_path)

# Output the data
print(data)
print(data.shape)