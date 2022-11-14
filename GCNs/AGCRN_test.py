import sys
sys.path.insert(0, '/home/bigdyl/topic-forecasting/network_analysis')

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

from preprocessed.utils import get_node_targets, get_node_features, get_edge_indices, get_edge_weights


class RecurrentGCN(torch.nn.Module):
    def __init__(self, number_of_nodes, in_channels, out_channels, embedding_dimensions):
        super(RecurrentGCN, self).__init__()
        self.recurrent = AGCRN(number_of_nodes=number_of_nodes,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               K=2,
                               embedding_dimensions=embedding_dimensions)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0


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
embedding_dimensions = 4
out_channels = 2

# node targets(label)
node_targets = get_node_targets(media, topic_num)
print('node targets: {}'.format(node_targets.shape))

number_of_nodes = node_targets[0].shape[0]
number_of_features = 1

# node features
node_features = get_node_features(media, topic_num)
# node_features = np.random.rand(18, number_of_nodes, number_of_features)
print('node feature: {}'.format(node_features.shape))

# edge indices and weights
edge_indices = get_edge_indices(media, topic_num)
edge_weights = get_edge_weights(media, topic_num)

dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

#########################
# 2. Model training
#########################
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)
model = RecurrentGCN(number_of_nodes=number_of_nodes,
                     in_channels=number_of_features,
                     out_channels=out_channels,
                     embedding_dimensions=embedding_dimensions)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
e = torch.empty(number_of_nodes, embedding_dimensions)
torch.nn.init.xavier_uniform_(e)
for epoch in tqdm(range(200)):
    cost = 0
    h = None
    for time, snapshot in enumerate(train_dataset):
        x = snapshot.x.view(1, number_of_nodes, number_of_features)  # (?, num of nodes, num of node features)
        y_hat, h = model(x, e, h)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

#########################
# 3. Model testing
#########################
model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    x = snapshot.x.view(1, number_of_nodes, number_of_features)
    y_hat, h = model(x, e, h)
    cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
cost = cost / (time + 1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
