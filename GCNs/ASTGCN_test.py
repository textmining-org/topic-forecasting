import numpy as np
import torch
from torch_geometric_temporal.nn.attention import ASTGCN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, DynamicGraphTemporalSignalBatch
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

from preprocessed.preprocessor import get_node_targets, get_node_features, get_edge_indices, get_edge_weights

#########################
# 1. Data preprocessing
# - networkx graph → DynamicGraphTemporalSignal
# - 기간 : 18개월 (2017-01 ~ 2018-06)
# - node : 600개
# - node features : 1개 (FIXME 현재 random value)
# - node targets : 1개 (word count)
# - edge indices : 36개 (시점 0)
# - edge weights : 36개 (시점 0, co-occurrence)
#########################


media = 'patents'
topic_num = 1

# node targets(label)
node_targets = get_node_targets(media, topic_num)
num_nodes = node_targets[0].shape[0]

# node features
node_features = get_node_features(media, topic_num)
num_features = node_features[0].shape[1]

# edge indices and weights
edge_indices = get_edge_indices(media, topic_num)
edge_weights = get_edge_weights(media, topic_num)

dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

#########################
# 2. Model training
#########################
model = ASTGCN(nb_block=2,
               in_channels=num_features,
               K=3,
               nb_chev_filter=64,
               nb_time_filter=64,
               time_strides=1,
               num_for_predict=1,
               len_input=64,
               num_of_vertices=num_nodes)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# FIXME input type and shape
model.train()
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        x = snapshot.x.view(1, num_nodes, num_features)
        y_hat = model(x, snapshot.edge_index)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
