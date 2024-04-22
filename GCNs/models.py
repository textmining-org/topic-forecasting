import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, AGCRN, A3TGCN, A3TGCN2, TGCN2
from torch.nn import DataParallel


class DCRNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K, out_size):
        super(DCRNNet, self).__init__()
        self.recurrent = DCRNN(in_channels, out_channels, K)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, edge_index, edge_weight):

        print(f'x shape: {x.shape}')
        print(f'edge_index type: {type(edge_index)}')
        print(f'edge_weight type: {type(edge_weight)}')
        # print(f'edge_index shape: {edge_index.shape}')

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
        self.node_features = node_features
        self.recurrent = CustomA3TGCN(node_features, out_channels, periods)
        self.linear = torch.nn.Linear(out_channels, out_size)

    def forward(self, x, edge_index, edge_weight):
        # feature 수 증가 시 RuntimeError 관련 shape 수정 : mat1 and mat2 shapes cannot be multiplied (50x1 and 2x32)
        # x = x.view(x.shape[0], 1, x.shape[1])
        x = x.view(x.shape[0], x.shape[1], 1)
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


class CustomA3TGCN2(A3TGCN2):
    def _setup_layers(self):
        self._base_tgcn = TGCN2(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=self.batch_size,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self._attention = torch.nn.Parameter(torch.empty(self.periods, device=device))
        self._attention = torch.nn.Parameter(torch.empty(self.periods))
        torch.nn.init.uniform_(self._attention)


class A3TGCNet2(torch.nn.Module):
    def __init__(self, node_features, out_channels, periods, batch_size):
        super(A3TGCNet2, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = CustomA3TGCN2(in_channels=node_features,
                                  out_channels=out_channels,
                                  periods=periods,
                                  batch_size=batch_size)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_channels, periods)

    def forward(self, x, edge_index, edge_weight):
        # feature 수 증가 시 RuntimeError 관련 shape 수정 : mat1 and mat2 shapes cannot be multiplied (50x1 and 2x32)
        # x = x.view(x.shape[0], 1, x.shape[1])
        # x = x.view(x.shape[0], x.shape[1], 1)
        h = self.tgnn(x, edge_index, edge_weight)
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
    elif args.model == 'a3tgcn2':
        return A3TGCNet2(node_features=num_features,
                         out_channels=args.out_channels,
                         periods=args.pred_len,
                         batch_size=args.batch_size)


def training(model, dataset, optimizer, criterion, num_features, num_nodes=None, embedd_dim=None):
    device = next(model.parameters()).device

    # TODO multi-processing
    if isinstance(model, DataParallel):
        model_class = model.module.__class__
    else:
        model_class = model.__class__

    if model_class == AGCRNet:
        h = None
        e = torch.empty(num_nodes, embedd_dim).to(device)
        torch.nn.init.xavier_uniform_(e)

    ys, y_hats = torch.Tensor().to(device), torch.Tensor().to(device)
    for time, snapshot in enumerate(dataset):
        optimizer.zero_grad()
        snapshot = snapshot.to(device)

        if time == 0:
            print(
                f'target: {snapshot.y.shape}, features: {snapshot.x.shape}, index: {snapshot.edge_index.shape}, attr: {snapshot.edge_attr.shape}')

        if model_class in [DCRNNet, TGCNet, A3TGCNet]:
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        elif model_class in [AGCRNet]:
            e = torch.empty(num_nodes, embedd_dim).to(device)
            torch.nn.init.xavier_uniform_(e)
            x = snapshot.x.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
            y_hat, h = model(x, e, h)
            h = h.detach()  # FIXME RuntimeError: Trying to backward through the graph a second time

        y_hat = y_hat.squeeze()

        loss = criterion(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()

        ys = torch.concat([ys, snapshot.y[None, :]], axis=0)
        y_hats = torch.concat([y_hats, y_hat[None, :]], axis=0)

    return y_hats, ys


def evaluating(model, dataset, num_features, num_nodes=None, embedd_dim=None):
    device = next(model.parameters()).device

    with torch.no_grad():
        if model.__class__ == AGCRNet:
            h = None
            e = torch.empty(num_nodes, embedd_dim).to(device)
            torch.nn.init.xavier_uniform_(e)

        ys, y_hats = torch.Tensor().to(device), torch.Tensor().to(device)
        for time, snapshot in enumerate(dataset):
            snapshot = snapshot.to(device)
            if model.__class__ in [DCRNNet, TGCNet, A3TGCNet]:
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            elif model.__class__ in [AGCRNet]:
                x = snapshot.x.view(1, num_nodes, num_features)
                y_hat, h = model(x, e, h)
            y_hat = y_hat.squeeze()

            ys = torch.concat([ys, snapshot.y[None, :]], axis=0)
            y_hats = torch.concat([y_hats, y_hat[None, :]], axis=0)

        return y_hats, ys


def training_loader(model, dataloader, edge_indices, edge_weights, optimizer, criterion, num_features, num_nodes,
                    embedd_dim):
    device = next(model.parameters()).device

    # print(f'in training loader... edge_indices len: {len(edge_indices)}')
    # print(f'in training loader... edge_indices 0 len: {len(edge_indices[0])}')
    # print(f'in training loader... edge_indices 0 0 shape: {len(edge_indices[0][0])}')
    # print(f'in training loader... edge_weights: {edge_weights}')

    print(f'edge_indices[0] length: {len(edge_indices[0])}')
    print(f'edge_indices[0] type: {type(edge_indices[0])}')

    print(f'edge_weights[0] length: {len(edge_weights[0])}')
    print(f'edge_weights[0] type: {type(edge_weights[0])}')

    # TODO multi-processing
    if isinstance(model, DataParallel):
        model_class = model.module.__class__
    else:
        model_class = model.__class__

    if model_class == AGCRNet:
        # FIXME dim, ex. (32, 50, 4, 12) : (time, node, features, seq_len)
        h = None
        e = torch.empty(num_nodes, embedd_dim).to(device)
        torch.nn.init.xavier_uniform_(e)

    edge_index = None
    edge_weight = None
    if model_class == A3TGCNet2:
        for i, edge_indices_b in enumerate(edge_indices):
            # print(edge_indices_b)
            # print(f'##### loader idx: {i}, edge_indices_b len: {len(edge_indices_b)}')
            for j, edge_indices_s in enumerate(edge_indices_b):
                # print(f'+++++{j} {len(edge_indices_s)}')
                for k, edge_indices_i in enumerate(edge_indices_s):
                    # print(f'-----{k} {edge_indices_i.shape}')
                    if edge_indices_i.shape[1] != 0:
                        # print(edge_indices_i)
                        edge_index = edge_indices_i
                        edge_weight = edge_weights[i][j][k]
        edge_index = torch.tensor(edge_index).to(device)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32).to(device)
    # else:
    #     edge_index = edge_indices
    #     edge_weight = edge_weights
    # print(f'in training loader... edge_index: {edge_index}')

    ys, y_hats = torch.Tensor().to(device), torch.Tensor().to(device)
    for idx, value in enumerate(dataloader):
        optimizer.zero_grad()
        node_features, node_targets = value

        y_hat = model(node_features, edge_index, edge_weight)

        if model_class in [A3TGCNet2]:
            y_hat = model(node_features, edge_index, edge_weight)
        elif model_class in [AGCRNet]:
            e = torch.empty(num_nodes, embedd_dim).to(device)
            torch.nn.init.xavier_uniform_(e)
            # FIXME dim, ex. (32, 50, 4, 12) : (time, node, features, seq_len)
            x = node_features.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
            y_hat, h = model(x, e, h)
            h = h.detach()  # FIXME RuntimeError: Trying to backward through the graph a second time

        # print(f'training - node features: {node_features.shape}\t'
        #       f'node targets: {node_targets.shape}\t'
        #       f'edge index: {edge_index.shape}\t'
        #       f'edge weight: {edge_weight.shape}\t'
        #       f'y_hat: {y_hat.shape}')

        loss = criterion(y_hat, node_targets)
        loss.backward()
        optimizer.step()

        ys = torch.concat([ys, node_targets], axis=0)
        y_hats = torch.concat([y_hats, y_hat], axis=0)

    return y_hats, ys


def evaluating_loader(model, dataloader, edge_indices, edge_weights, num_features, num_nodes, embedd_dim):
    device = next(model.parameters()).device

    with torch.no_grad():

        if model.__class__ == AGCRNet:
            h = None
            e = torch.empty(num_nodes, embedd_dim).to(device)
            torch.nn.init.xavier_uniform_(e)

        if model.__class__ == A3TGCNet2:
            edge_index = None
            edge_weight = None
            for i, edge_indices_b in enumerate(edge_indices):
                # print(edge_indices_b)
                # print(f'##### loader idx: {i}, edge_indices_b len: {len(edge_indices_b)}')
                for j, edge_indices_s in enumerate(edge_indices_b):
                    # print(f'+++++{j} {len(edge_indices_s)}')
                    for k, edge_indices_i in enumerate(edge_indices_s):
                        # print(f'-----{k} {edge_indices_i.shape}')

                        edge_indices_i_shape = edge_indices_i.shape if edge_indices_i.shape[1] != 0 else None
                        edge_weight_shape = edge_weights[i][j][k].shape if edge_indices_i.shape[1] != 0 else None
                        # print(f'evaluating - iter({i},{j},{k})\t'
                        #       f'edge indices: {len(edge_indices)}\t'
                        #       f'edge indices b: {len(edge_indices_b)}\t'
                        #       f'edge index: {edge_indices_i_shape}\t'
                        #       f'edge weights: {len(edge_weights)}\t'
                        #       f'edge weight: {edge_weight_shape}')

                        if edge_indices_i.shape[1] != 0:
                            # print(edge_indices_i)
                            # print(f'{edge_weights[i][j][k]}')
                            edge_index = edge_indices_i
                            edge_weight = edge_weights[i][j][k]
            edge_index = torch.tensor(edge_index).to(device)
            edge_weight = torch.tensor(edge_weight, dtype=torch.float32).to(device)

        ys, y_hats, xs = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)
        for time, value in enumerate(dataloader):
            node_features, node_targets = value
            # print(node_features.shape, node_targets.shape)
            if model.__class__ in [A3TGCNet2]:
                y_hat = model(node_features, edge_index, edge_weight)
            elif model.__class__ in [AGCRNet]:
                x = node_features.view(1, num_nodes, num_features)
                y_hat, h = model(x, e, h)

            ys = torch.concat([ys, node_targets], axis=0)
            y_hats = torch.concat([y_hats, y_hat], axis=0)

            xs = torch.concat([xs, node_features[:, :, -1:, :]], axis=0)

        return y_hats, ys, xs
