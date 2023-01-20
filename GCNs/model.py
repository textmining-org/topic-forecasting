import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, AGCRN, A3TGCN


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

class CustomA3TGCN(A3TGCN):
    def _setup_layers(self):
        self._base_tgcn = TGCN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
        )
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        self._attention = torch.nn.Parameter(torch.empty(self.periods))
        torch.nn.init.uniform_(self._attention)

class A3TGCNet(torch.nn.Module):
    def __init__(self, node_features, out_channels, periods, out_size):
        super(A3TGCNet, self).__init__()
        self.recurrent = CustomA3TGCN(node_features, out_channels, periods)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h

def get_model(args, num_nodes, num_features):
    if args.model == 'dcrnn':
        return DCRNNet(in_channels=num_features,
                       out_channels=args.out_channels,
                       K=args.K,
                       out_size=args.out_size)
    elif args.model == 'tgcn':
        return TGCNet(in_channels=num_features,
                      out_channels=args.out_channels,
                      out_size=args.out_size)
    elif args.model == 'agcrn':
        return AGCRNet(number_of_nodes=num_nodes,
                       in_channels=num_features,
                       out_channels=args.out_channels,
                       K=args.K,
                       embedding_dimensions=args.embedd_dim,
                       out_size=args.out_size)
    elif args.model == 'a3tgcn':
        return A3TGCNet(node_features=num_features,
                        out_channels=args.out_channels,
                        periods=args.periods,
                        out_size=args.out_size)