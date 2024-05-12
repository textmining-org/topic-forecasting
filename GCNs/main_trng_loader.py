import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import torch
from tqdm import tqdm
from _commons.utils import save_metrics, fix_randomness, EarlyStopping, exists_metrics, adjust_learning_rate
from config import get_config
from models import get_model, training_loader, evaluating_loader
from preprocessed.preprocessor_for_loader import get_loader
import math

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    model_save_path = os.path.join(results_path, 'models')
    node_feature_type = '_'.join(args.node_feature_type)
    model_filename = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}.pt'
    metric_save_path = os.path.join(results_path, args.metrics_file)

    arg_names = ['media', 'model', 'node_feature_type', 'epochs', 'batch_size', 'lr', 'num_train_clusters',
                 'num_valid_clusters', 'num_test_clusters', 'seq_len', 'pred_len', 'desc']
    if args.model == 'a3tgcn2':
        arg_names.extend(['out_channels'])
    elif args.model == 'agcrn':
        arg_names.extend(['K', 'embedd_dim', 'out_channels'])
    elif args.model in ['lstm', 'gru']:
        arg_names.extend(['hidden_size', 'num_layers'])

    if exists_metrics(metric_save_path, args, arg_names):
        print(f'There exist experiments results! - {args}')
        sys.exit()

    # fix randomness
    fix_randomness(args.seed)

    gpu = 'cuda:' + args.device
    device = torch.device(gpu)

    # load data
    train_cluster_dirs = [os.path.join(args.cluster_dir + '/train_valid', i)
                          for i in os.listdir(args.cluster_dir + '/train_valid')][:args.num_train_clusters]
    # validation set 고정: 8000~9999
    valid_cluster_dirs = [os.path.join(args.cluster_dir + '/train_valid', i)
                          for i in os.listdir(args.cluster_dir + '/train_valid')][8000:8000 + args.num_valid_clusters]
    test_cluster_dirs = [os.path.join(args.cluster_dir + '/test', i)
                         for i in os.listdir(args.cluster_dir + '/test')][:args.num_test_clusters]

    train_dataloader_packages = []
    print('##### read train clusters')
    for _c_dir in train_cluster_dirs:
        dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar, max_val_tar, eps = get_loader(
            _c_dir,
            node_feature_type=args.node_feature_type,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            device=device)

        train_dataloader_packages.append(
            [dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar,
             max_val_tar, eps])
        # print(len(dataloader))

    valid_dataloader_packages = []
    print('##### read valid clusters')
    for _c_dir in valid_cluster_dirs:
        dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar, max_val_tar, eps = get_loader(
            _c_dir,
            node_feature_type=args.node_feature_type,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            device=device)
        valid_dataloader_packages.append(
            [dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar,
             max_val_tar, eps])

    test_dataloader_packages = []
    print('##### read test clusters')
    for _c_dir in test_cluster_dirs:
        dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar, max_val_tar, eps = get_loader(
            _c_dir,
            node_feature_type=args.node_feature_type,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            batch_size=args.batch_size,
            device=device)
        test_dataloader_packages.append(
            [dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar,
             max_val_tar, eps])

    print(
        f'##### train len: {len(train_dataloader_packages)}, valid len: {len(valid_dataloader_packages)}, test len: {len(test_dataloader_packages)}')

    # FIXME data parallel : multi-gpu
    # create GCN model
    model = get_model(args, num_nodes, num_features)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')
    early_stopping = EarlyStopping(patience=args.patience)

    # mse_log = {}
    best_epoch, best_train_mse, best_valid_mse, best_test_mse = 0, math.inf, math.inf, math.inf
    for epoch in tqdm(range(args.epochs)):
        print(f'##### epoch: {epoch}')
        train_y_hats, train_ys, valid_y_hats, valid_ys, test_y_hats, test_ys \
            = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(
            device), torch.Tensor().to(device), torch.Tensor().to(device)
        for _ds_idx, _curr_dataset_pack in enumerate(train_dataloader_packages):
            # training
            model.train()
            train_dataloader = _curr_dataset_pack[0]
            edge_indices = _curr_dataset_pack[1]
            edge_weights = _curr_dataset_pack[2]
            train_y_hat, train_y = training_loader(model, train_dataloader, edge_indices, edge_weights,
                                                   optimizer, criterion, num_features, num_nodes, args.embedd_dim)
            train_y_hats = torch.concat([train_y_hats, train_y_hat[None, :]])
            train_ys = torch.concat([train_ys, train_y[None, :]])

        for _ds_idx, _curr_dataset_pack in enumerate(valid_dataloader_packages):
            # print(f'valid _ds_idx - {_ds_idx}')
            # evaluating
            model.eval()
            valid_dataloader = _curr_dataset_pack[0]
            edge_indices = _curr_dataset_pack[1]
            edge_weights = _curr_dataset_pack[2]
            valid_y_hat, valid_y, _ = evaluating_loader(model, valid_dataloader, edge_indices, edge_weights,
                                                        num_features,
                                                        num_nodes, args.embedd_dim)
            valid_y_hats = torch.concat([valid_y_hats, valid_y_hat[None, :]])
            valid_ys = torch.concat([valid_ys, valid_y[None, :]])

        for _ds_idx, _curr_dataset_pack in enumerate(test_dataloader_packages):
            # evaluating
            model.eval()
            test_dataloader = _curr_dataset_pack[0]
            edge_indices = _curr_dataset_pack[1]
            edge_weights = _curr_dataset_pack[2]
            test_y_hat, test_y, _ = evaluating_loader(model, test_dataloader, edge_indices, edge_weights, num_features,
                                                      num_nodes, args.embedd_dim)
            test_y_hats = torch.concat([test_y_hats, test_y_hat[None, :]])
            test_ys = torch.concat([test_ys, test_y[None, :]])

        MSE = torch.nn.MSELoss(reduction='mean')
        MAE = torch.nn.L1Loss(reduction='mean')

        train_mse, train_mae = MSE(train_y_hats, train_ys).item(), MAE(train_y_hats, train_ys).item()
        valid_mse, valid_mae = MSE(valid_y_hats, valid_ys).item(), MAE(valid_y_hats, valid_ys).item()
        test_mse, test_mae = MSE(test_y_hats, test_ys).item(), MAE(test_y_hats, test_ys).item()
        print('[Epoch: {:3} (MSE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f}'.format(epoch, train_mse, valid_mse,
                                                                                          test_mse))
        print('[Epoch: {:3} (MAE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f}'.format(epoch, train_mae, valid_mae,
                                                                                          test_mae))
        if best_valid_mse > valid_mse:
            best_epoch = epoch
            best_train_mse, best_valid_mse, best_test_mse = train_mse, valid_mse, test_mse
            best_train_mae, best_valid_mae, best_test_mae = train_mae, valid_mae, test_mae
            print("- Best (MSE) update!! Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
                best_train_mse, best_valid_mse, best_test_mse, best_epoch))
            print("- Best (MAE) update!! Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
                best_train_mae, best_valid_mae, best_test_mae, best_epoch))
            # save best model
            torch.save(model.state_dict(), os.path.join(model_save_path, model_filename))

        # early stopping
        early_stopping(valid_mse)
        if early_stopping.early_stop:
            print("##### Early stopping ...")
            break

        adjust_learning_rate(optimizer, epoch + 1, args.lr, '1')

    print("[Final (BEST MSE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
        best_train_mse, best_valid_mse, best_test_mse, best_epoch))
    print("[Final (BEST MAE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
        best_train_mae, best_valid_mae, best_test_mae, best_epoch, best_epoch))

    metric_names = ['mse', 'mae']
    metrics = [best_test_mse, best_test_mae]
    save_metrics(metric_save_path, args, arg_names, metrics, metric_names)
