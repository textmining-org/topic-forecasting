import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import AGCRN
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import TGCN


class DCRNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, out_size):
        super(DCRNNet, self).__init__()
        self.recurrent = DCRNN(in_channels, out_channels, K)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class TGCNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super(TGCNet, self).__init__()
        self.recurrent = TGCN(in_channels, out_channels)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class AGCRNet(torch.nn.Module):
    def __init__(self, number_of_nodes, in_channels, out_channels, K, embedding_dimensions, out_size):
        super(AGCRNet, self).__init__()
        self.recurrent = AGCRN(number_of_nodes=number_of_nodes,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               K=K,
                               embedding_dimensions=embedding_dimensions)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0
