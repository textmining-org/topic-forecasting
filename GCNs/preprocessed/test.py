import numpy as np

loc_node_targets = './node_targets.npy'
node_targets = np.load(loc_node_targets)
print(node_targets.shape)

loc_node_targets = './patents/4.topic/1/word_count.node_targets.npy'
node_targets = np.load(loc_node_targets)
print(node_targets.shape)
print(node_targets)

