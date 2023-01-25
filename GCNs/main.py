import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
# from _commons.loss import MSE, MAE
from _commons.utils import save_metrics, fix_randomness
from config import get_config
from model import get_model
from preprocessed.preprocessor import get_dataset


def training(model_name, model, dataset, optimizer, criterion):
    device = next(model.parameters()).device
    h = None

    ys, y_hats = torch.Tensor().to(device), torch.Tensor().to(device)
    for time, snapshot in enumerate(dataset):
        optimizer.zero_grad()
        snapshot = snapshot.to(device)
        if model_name in ['dcrnn', 'tgcn', 'a3tgcn']:
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        elif model_name in ['agcrn']:
            x = snapshot.x.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
            y_hat, h = model(x, e, h)
        y_hat = y_hat.squeeze()

        loss = criterion(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()

        ys = torch.concat([ys, snapshot.y[None, :]], axis=0)
        y_hats = torch.concat([y_hats, y_hat[None, :]], axis=0)

    return y_hats, ys


def evaluating(model_name, model, dataset):
    device = next(model.parameters()).device
    with torch.no_grad():
        ys, y_hats = torch.Tensor().to(device), torch.Tensor().to(device)
        for time, snapshot in enumerate(dataset):
            snapshot = snapshot.to(device)
            if model_name in ['dcrnn', 'tgcn', 'a3tgcn']:
                y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            elif model_name in ['agcrn']:
                x = snapshot.x.view(1, num_nodes, num_features)
                y_hat, h = model(x, e, h)
            y_hat = y_hat.squeeze()

            ys = torch.concat([ys, snapshot.y[None, :]], axis=0)
            y_hats = torch.concat([y_hats, y_hat[None, :]], axis=0)

        return y_hats, ys


if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    tb_train_loss = SummaryWriter(log_dir=f'../_tensorboard/{args.node_feature_type}/{args.model}/loss')

    # fix randomness
    fix_randomness(args.seed)

    gpu = 'cuda:' + args.device
    device = torch.device(gpu)

    # load data
    refine_data = True if args.model == 'dcrnn' else False

    # TODO WARNING - check the location of cluster dirs
    cluster_dirs = [os.path.join(args.data_dir, i) for i in os.listdir(args.data_dir)][:100]
    dataset_packages = []
    for _c_dir in cluster_dirs:
        dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
            = get_dataset(_c_dir, args.node_feature_type, args.discard_index, refine_data)
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
        dataset_packages.append([train_dataset, test_dataset,
                                 num_nodes, num_features, min_val_tar, max_val_tar, eps])

    # create GCN model
    model = get_model(args, num_nodes, num_features)
    model.to(device)
    if args.model == 'agcrn':
        e = torch.empty(num_nodes, args.embedd_dim)
        torch.nn.init.xavier_uniform_(e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    mse_log = {}
    for epoch in tqdm(range(args.epochs)):

        train_y_hats, train_ys, test_y_hats, test_ys = torch.Tensor().to(device), torch.Tensor().to(
            device), torch.Tensor().to(device), torch.Tensor().to(device)
        for _ds_idx, _curr_dataset_pack in enumerate(dataset_packages):
            model.train()
            train_dataset = _curr_dataset_pack[0]
            train_y_hat, train_y = training(args.model, model, train_dataset, optimizer, criterion)

            train_y_hats = torch.concat([train_y_hats, train_y_hat])
            train_ys = torch.concat([train_ys, train_y])

            model.eval()
            test_dataset = _curr_dataset_pack[1]
            test_y_hat, test_y = evaluating(args.model, model, test_dataset)

            test_y_hats = torch.concat([test_y_hats, test_y_hat])
            test_ys = torch.concat([test_ys, test_y])

        MSE = torch.nn.MSELoss(reduction='mean')
        MAE = torch.nn.L1Loss(reduction='mean')

        train_mse, train_mae = MSE(train_y_hats, train_ys), MAE(train_y_hats, train_ys)
        test_mse, test_mae = MSE(test_y_hats, test_ys), MAE(test_y_hats, test_ys)

        print(train_mse)
        mse_log[epoch] = train_mse

    test_y_hats = test_y_hats.detach().cpu()
    test_ys = test_ys.detach().cpu()

    # save_metrics
    results_path = os.path.abspath(args.results_path)
    arg_names = ['data_dir', 'node_feature_type', 'model', 'epochs', 'lr']
    metrics = [test_mse.detach().cpu().numpy(), test_mae.detach().cpu().numpy()]
    metric_names = ['mse', 'mae']
    save_metrics(results_path, args, arg_names, metrics, metric_names)

    # FIXME bug : out of index
    # de-normalizing
    # denorm_y_hats = [denormalizer(ds_y_hat, dataset_packages[ds_idx][4], dataset_packages[ds_idx][5], dataset_packages[ds_idx][6]) for
    #                  ds_idx, ds_y_hat in enumerate(test_y_hats)]
    # denorm_ys = [denormalizer(ds_y_hat, dataset_packages[ds_idx][4], dataset_packages[ds_idx][5], dataset_packages[ds_idx][6]) for
    #              ds_idx, ds_y_hat in enumerate(test_ys)]
    #
    # save_pred_y(results_path, args.data_dir, args.topic_num, args.model, denorm_ys, denorm_y_hats)
