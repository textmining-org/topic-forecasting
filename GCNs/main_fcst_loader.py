import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import numpy as np
import torch

from _commons.utils import save_metrics, save_fsct_y, fix_randomness, exists_metrics
from config import get_config
from models import get_model, evaluating_loader
from preprocessed.preprocessor_for_loader import get_loader, denormalizer

import natsort

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    # fix randomness
    fix_randomness(args.seed)

    gpu = 'cuda:' + args.device
    device = torch.device(gpu)

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    model_save_path = os.path.join(results_path, 'models')
    node_feature_type = '_'.join(args.node_feature_type)
    metric_save_path = os.path.join(results_path, 'metrics_fcst.csv')
    model_filename = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}.pt'

    arg_names = ['media', 'model', 'node_feature_type', 'epochs', 'batch_size', 'lr', 'num_train_clusters',
                 'num_valid_clusters', 'num_test_clusters', 'seq_len', 'pred_len', 'desc']
    # if exists_metrics(metric_save_path, args, arg_names):
    #     print(f'There exist experiments results! - {args}')
    #     sys.exit()

    # load data
    topic_dirs = [os.path.join(args.topic_dir, i) for i in natsort.natsorted(os.listdir(args.topic_dir))]
    print(topic_dirs)

    topic_dataloader_packages = []
    for _t_dir in topic_dirs:
        dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar, max_val_tar, eps \
            = get_loader(_t_dir,
                         node_feature_type=args.node_feature_type,
                         seq_len=args.seq_len,
                         pred_len=args.pred_len,
                         batch_size=args.batch_size,
                         device=device)
        topic_dataloader_packages.append(
            [dataloader, edge_indices, edge_weights, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar,
             max_val_tar, eps])

    # load model
    model = get_model(args, num_nodes, num_features)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_filename)))
    model.eval()
    print(model)

    y_hats, ys, xs = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device)
    for _ds_idx, _curr_dataset_pack in enumerate(topic_dataloader_packages):
        dataloader = _curr_dataset_pack[0]
        edge_indices = _curr_dataset_pack[1]
        edge_weights = _curr_dataset_pack[2]
        y_hat, y, x = evaluating_loader(model, dataloader, edge_indices, edge_weights, num_features, num_nodes,
                                        args.embedd_dim)
        x = x.squeeze()
        print(_ds_idx, len(dataloader), y_hat.shape, y.shape, x.shape)

        y_hats = torch.concat([y_hats, y_hat[None, :]])
        ys = torch.concat([ys, y[None, :]])
        xs = torch.concat([xs, x[None, :]])
        print(_ds_idx, len(dataloader), y_hats.shape, ys.shape, xs.shape)

    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    mse, mae = MSE(y_hats, ys).item(), MAE(y_hats, ys).item()

    # save forecasting metrics
    arg_names = ['media', 'model', 'node_feature_type', 'epochs', 'batch_size', 'lr', 'num_train_clusters',
                 'num_valid_clusters', 'num_test_clusters', 'seq_len', 'pred_len', 'desc']
    metric_names = ['mse', 'mae']
    metrics = [mse, mae]
    save_metrics(metric_save_path, args, arg_names, metrics, metric_names)
    print(f'MSE: {mse}, MAE: {mae}')

    # de-normalizing
    y_hats_denorm = \
        [denormalizer(ds_y_hat.detach().cpu(), topic_dataloader_packages[ds_idx][7],
                      topic_dataloader_packages[ds_idx][8],
                      topic_dataloader_packages[ds_idx][9]) for ds_idx, ds_y_hat in enumerate(y_hats)]
    ys_denorm = [
        denormalizer(ds_y.detach().cpu(), topic_dataloader_packages[ds_idx][7],
                     topic_dataloader_packages[ds_idx][8],
                     topic_dataloader_packages[ds_idx][9]) for ds_idx, ds_y in enumerate(ys)]
    xs_denorm = [
        denormalizer(ds_x.detach().cpu(), topic_dataloader_packages[ds_idx][5],
                     topic_dataloader_packages[ds_idx][6],
                     topic_dataloader_packages[ds_idx][9]) for ds_idx, ds_x in enumerate(xs)]
    y_hats_denorm = np.stack(y_hats_denorm, axis=0)
    ys_denorm = np.stack(ys_denorm, axis=0)
    xs_denorm = np.stack(xs_denorm, axis=0)

    print(y_hats_denorm.shape, ys_denorm.shape, xs_denorm.shape)

    # save forecasting values
    save_fsct_y(fcst_val_save_path, args, ys_denorm, y_hats_denorm, xs_denorm)
