import os

import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

from _commons.loss import MSE, MAE
from _commons.utils import save_metrics, fix_randomness, save_pred_y
from config import get_config
from model import get_model
from preprocessed.preprocessor import denormalizer, get_dataset
from torch.utils.data import DataLoader

def training(model_name, model, dataset, optimizer, criterion):
    h = None

    for time, snapshot in enumerate(dataset):
        optimizer.zero_grad()

        if model_name in ['dcrnn', 'tgcn', 'a3tgcn']:
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        elif model_name in ['agcrn']:
            x = snapshot.x.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
            y_hat, h = model(x, e, h)
        y_hat = y_hat.squeeze()

        loss = criterion(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()

    loss_value = criterion(y_hat, snapshot.y)

    return y_hat, snapshot.y, loss_value

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    tb_train_loss = SummaryWriter(log_dir=f'../_tensorboard/{args.media}/{args.topic_num}/{args.model}/loss')

    # fix randomness
    fix_randomness(args.seed)

    # load data
    refine_data = True if args.model == 'dcrnn' else False
    dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
        = get_dataset(args.media, args.topic_num, args.discard_index, refine_data)
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # create GCN model
    model = get_model(args, num_nodes, num_features)
    if args.model == 'agcrn':
        e = torch.empty(num_nodes, args.embedd_dim)
        torch.nn.init.xavier_uniform_(e)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    # training
    model.train()
    for epoch in tqdm(range(args.epochs)):
        y_hat, y, mse = training(args.model,model, train_dataset, optimizer, criterion)
        print(f'{epoch}-th epochs, train mse: {mse}')
        # cost = 0
        # h = None
        #
        # # for time, snapshot in enumerate(train_loader):
        # for time, snapshot in enumerate(train_dataset):
        #     if args.model == 'dcrnn' or args.model == 'tgcn' or args.model == 'a3tgcn':
        #         y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        #     elif args.model == 'agcrn':
        #         x = snapshot.x.view(1, num_nodes, num_features)  # (?, num of nodes, num of node features)
        #         y_hat, h = model(x, e, h)
        #
        #     cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
        # cost = cost / (time + 1)
        # cost.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        tb_train_loss.add_scalar(f"{args.media} / {args.topic_num} / {args.model} / Loss: mse",
                                 mse, epoch)

    # evaluating
    model.eval()
    # TODO y_hats, ys = torch.Tensor(), torch.Tensor()
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

    y_hats = torch.stack(y_hats).detach().numpy()
    ys = torch.stack(ys).detach().numpy()

    mse = MSE(y_hats, ys)
    print("MSE: {:.4f}".format(mse))

    mae = MAE(y_hats, ys)
    print("MAE: {:.4f}".format(mae))

    # save_metrics
    results_path = os.path.abspath(args.results_path)
    arg_names = ['media', 'topic_num', 'model', 'epochs', 'lr', 'discard_index']
    metrics = [mse, mae]
    metric_names = ['mse', 'mae']
    save_metrics(results_path, args, arg_names, metrics, metric_names)

    # de-normalizing
    y_hat = denormalizer(y_hat.detach().numpy(), min_val_tar, max_val_tar, eps)
    ys = denormalizer(ys.detach().numpy(), min_val_tar, max_val_tar, eps)
    # save ground truth and forecasting results
    save_pred_y(results_path, args.media, args.topic_num, args.model, ys, y_hat)

    tb_train_loss.flush()
    tb_train_loss.close()
