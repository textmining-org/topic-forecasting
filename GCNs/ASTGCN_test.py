from torch_geometric_temporal.nn.attention import ASTGCN
from tqdm import tqdm
import numpy as np
from preprocessed.utils import get_node_targets, get_edge_indices_and_weights
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import torch.nn.functional as F


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

node_targets = get_node_targets()
print('node targets: {}'.format(node_targets.shape))

num_of_nodes = node_targets[0].shape[0]
num_of_node_features = 1

# FIXME 추후 centrality 등으로 변경 필요
node_features = np.random.rand(18, num_of_nodes, num_of_node_features)
print('node feature: {}'.format(node_features.shape))

dict_edge_indices, dict_edge_weights = get_edge_indices_and_weights()
with open('./preprocessed/edge_attributes.txt', 'r') as f_eattr:
    edge_attrs = f_eattr.read().splitlines()

    edge_indices = []
    edge_weights = []
    for edge_attr in edge_attrs:
        if edge_attr in dict_edge_indices and edge_attr in dict_edge_weights:
            edge_indices.append(np.array(dict_edge_indices[edge_attr]))
            edge_weights.append(np.array(dict_edge_weights[edge_attr]))

    print('edge indices: {} * {}'.format(len(edge_indices), edge_indices[0].shape))
    print('edge weights: {} * {}'.format(len(edge_weights), edge_weights[0].shape))

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

#########################
# 2. Model training
#########################
model = ASTGCN(nb_block=2,
               in_channels=1,
               K=3,
               nb_chev_filter=64,
               nb_time_filter=64,
               time_strides=1,
               num_for_predict=1,
               len_input=18,
               num_of_vertices=600)

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# FIXME input type and shape
model.train()
for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()