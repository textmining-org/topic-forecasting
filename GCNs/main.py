import torch.backends.cudnn as cudnn
import random
import torch
from models import DCRNNet, TGCNet, AGCRNet, A3TGCNet
import numpy as np
from config import get_config
from preprocessed.utils import get_node_targets, get_node_features, get_edge_indices, get_edge_weights, \
    refine_graph_data, normalizer, denormalizer
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime


def get_model(model_name):
    if model_name == 'dcrnn':
        return DCRNNet
    elif model_name == 'tgcn':
        return TGCNet
    elif model_name == 'agcrn':
        return AGCRNet
    elif model_name == 'a3tgcn':
        return A3TGCNet


def get_dataset(media, topic_num, discard_index, refine_data=False):
    # node targets(label)
    node_targets = get_node_targets(media=media, topic_num=topic_num, discard_index=discard_index)
    node_targets, min_val_tar, max_val_tar, eps = normalizer(node_targets)
    num_nodes = node_targets[0].shape[0]

    # node features
    node_features = get_node_features(media=media, topic_num=topic_num, discard_index=discard_index)
    node_features, min_val_fea, max_val_fea, eps = normalizer(node_features)
    num_features = node_features[0].shape[1]

    # edge indices and weights
    edge_indices = get_edge_indices(media=media, topic_num=topic_num, discard_index=discard_index)
    edge_weights = get_edge_weights(media=media, topic_num=topic_num, discard_index=discard_index)

    if refine_data == True:
        node_targets, node_features, edge_indices, edge_weights = refine_graph_data(node_targets, node_features,
                                                                                    edge_indices, edge_weights)
    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

    return dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps


def save_results(results_path, args, mse, mae):
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path, index_col=0)
    else:
        columns = ['timestamp', 'media', 'topic_num', 'model', 'mse', 'mae', 'epochs', 'lr', 'discard_index']
        df_results = pd.DataFrame(columns=columns)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df_results.loc[len(df_results)] = [timestamp, args.media, args.topic_num, args.model, mse, mae, args.epochs,
                                       args.lr, args.discard_index]
    df_results.sort_values(by="mse", ascending=True, inplace=True)
    df_results.to_csv(results_path)


def exists_results(results_path, args):
    if not os.path.exists(results_path):
        return False

    df_results = pd.read_csv(results_path)

    for index, result in df_results.iterrows():
        if result['model'] == args.model and result['media'] == args.media and \
                result['topic_num'] == args.topic_num and \
                result['epochs'] == args.epochs and result['lr'] == args.lr and \
                result['discard_index'] == args.discard_index:
            return True
    return False


if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    # fix randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    # load data
    refine_data = True if args.model == 'dcrnn' else False

    dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps = get_dataset(args.media, args.topic_num,
                                                                                  args.discard_index, refine_data)
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
    elif args.model == 'a3tgcn':
        model = model(node_features=num_features,
                      out_channels=args.out_channels,
                      periods=args.periods,
                      out_size=args.out_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in tqdm(range(args.epochs)):
        cost = 0
        h = None
        for time, snapshot in enumerate(train_dataset):
            if args.model == 'dcrnn' or args.model == 'tgcn' or args.model == 'a3tgcn':
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
    # y_hats, ys = torch.Tensor(), torch.Tensor()
    y_hats, ys = [], []
    print(test_dataset)
    for time, snapshot in enumerate(test_dataset):
        if args.model == 'dcrnn' or args.model == 'tgcn':
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        elif args.model == 'agcrn':
            x = snapshot.x.view(1, num_nodes, num_features)
            y_hat, h = model(x, e, h)

        y_hats.append(y_hat.squeeze())
        ys.append(snapshot.y)

    y_hats = torch.stack(y_hats)
    ys = torch.stack(ys)
    print(y_hats.shape)
    print(ys.shape)

    mse_loss = torch.nn.MSELoss(reduction='mean')
    mse_value = mse_loss(y_hats, ys)
    print("MSE: {:.4f}".format(mse_value.item()))

    mae_loss = torch.nn.L1Loss(reduction='mean')
    mae_value = mae_loss(y_hats, ys)
    print("MAE: {:.4f}".format(mae_value.item()))

    # save_results
    results_path = os.path.abspath(f'./results/results.csv')
    save_results(results_path, args, mse_value.item(), mae_value.item())

    y_hat = denormalizer(y_hat.detach().numpy(), min_val_tar, max_val_tar, eps)
    ys = denormalizer(ys.detach().numpy(), min_val_tar, max_val_tar, eps)

    np.save(f"./results/{args.media}_{args.topic_num}_{args.model}_pred.npy", y_hat)
    np.save(f"./results/{args.media}_{args.topic_num}_{args.model}_true.npy", ys)
