import torch.backends.cudnn as cudnn
import random
import torch
from models import DCRNNet, TGCNet, AGCRNet
import numpy as np
from config import get_config
from preprocessed.utils import get_node_targets, get_node_features, get_edge_indices, get_edge_weights, normalizer
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm


def get_model(model_name):
    if model_name == 'dcrnn':
        return DCRNNet
    elif model_name == 'tgcn':
        return TGCNet
    elif model_name == 'agcrn':
        return AGCRNet


def get_dataset(media, topic_num, discard_index):
    # node targets(label)
    node_targets = get_node_targets(media=media, topic_num=topic_num, discard_index=discard_index)
    node_targets, min_val_tar, max_val_tar = normalizer(node_targets)
    num_nodes = node_targets[0].shape[0]

    # node features
    node_features = get_node_features(media=media, topic_num=topic_num, discard_index=discard_index)
    node_features, min_val_fea, max_val_fea = normalizer(node_features)
    num_features = node_features[0].shape[1]

    # edge indices and weights
    edge_indices = get_edge_indices(media=media, topic_num=topic_num, discard_index=discard_index)
    edge_weights = get_edge_weights(media=media, topic_num=topic_num, discard_index=discard_index)

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

    return dataset, num_nodes, num_features


if __name__ == "__main__":
    # set configuration
    args = get_config()

    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    # load data
    dataset, num_nodes, num_features = get_dataset(args.media, args.topic_num, args.discard_index)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

    # create GCN model
    model = get_model(args.model)
    if args.model == 'dcrnn':
        model = model(in_channels=num_features,
                      out_channels=args.out_channels,
                      K=args.K,
                      out_size=args.out_size)
    elif args.model == 'tgcn':
        model = model(in_channels=num_features,
                      out_channels=args.out_channels,
                      out_size=args.out_size)
    elif args.model == 'agcrn':
        model = model(number_of_nodes=num_nodes,
                      in_channels=num_features,
                      out_channels=args.out_channels,
                      K=args.K,
                      embedding_dimensions=args.embedd_dim,
                      out_size=args.out_size)
        e = torch.empty(num_nodes, args.embedd_dim)
        torch.nn.init.xavier_uniform_(e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in tqdm(range(200)):
        cost = 0
        h = None
        for time, snapshot in enumerate(train_dataset):
            if args.model == 'dcrnn' or args.model == 'tgcn':
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            elif args.model == 'agcrn':
                x = snapshot.x.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
                y_hat, h = model(x, e, h)

            cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        cost = cost / (time + 1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    y_hats, ys = torch.Tensor(), torch.Tensor()
    for time, snapshot in enumerate(test_dataset):
        if args.model == 'dcrnn' or args.model == 'tgcn':
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        elif args.model == 'agcrn':
            x = snapshot.x.view(1, num_nodes, num_features)
            y_hat, h = model(x, e, h)

        y_hats = torch.concat([y_hats, y_hat.squeeze()])
        ys = torch.concat([ys, snapshot.y])

    mse_loss = torch.nn.MSELoss(reduction='mean')
    mse_value = mse_loss(y_hats, ys)
    print("MSE: {:.4f}".format(mse_value.item()))

    mae_loss = torch.nn.L1Loss(reduction='mean')
    mae_value = mae_loss(y_hats, ys)
    print("MAE: {:.4f}".format(mae_value.item()))
