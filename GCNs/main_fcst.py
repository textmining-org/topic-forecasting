import os

import numpy as np
import torch

from _commons.utils import save_metrics, save_fsct_y
from config import get_config
from models import get_model, evaluating
from preprocessed.preprocessor import get_dataset, denormalizer

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    model_save_path = os.path.join(results_path, 'models')
    model_filename = args.model + '_' + '_'.join(args.node_feature_type) + '.pt'

    # load data
    refine_data = True if args.model == 'dcrnn' else False
    topic_dirs = [os.path.join(args.topic_dir, i) for i in os.listdir(args.topic_dir)]
    topic_dataset_packages = []
    for _t_dir in topic_dirs:
        topic_dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
            = get_dataset(_t_dir, args.node_feature_type, args.discard_index, refine_data)
        topic_dataset_packages.append([topic_dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps])

    # load model
    model = get_model(args, num_nodes, num_features)
    gpu = 'cuda:' + args.device
    device = torch.device(gpu)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_filename)))
    model.eval()

    y_hats, ys = torch.Tensor().to(device), torch.Tensor().to(device)
    for _ds_idx, _curr_dataset_pack in enumerate(topic_dataset_packages):
        dataset = _curr_dataset_pack[0]
        y_hat, y = evaluating(model, dataset, num_features, num_nodes, args.embedd_dim)

        y_hats = torch.concat([y_hats, y_hat[None, :, :]])
        ys = torch.concat([ys, y[None, :, :]])

    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    mse, mae = MSE(y_hats, ys).item(), MAE(y_hats, ys).item()

    # save forecasting metrics
    arg_names = ['model', 'node_feature_type', 'topic_dir']
    metric_names = ['mse', 'mae']
    metrics = [mse, mae]
    save_metrics(results_path, 'metrics_fcst.csv', args, arg_names, metrics, metric_names)

    # de-normalizing
    y_hats_denorm = \
        [denormalizer(ds_y_hat.detach().cpu(), topic_dataset_packages[ds_idx][3], topic_dataset_packages[ds_idx][4],
                      topic_dataset_packages[ds_idx][5]) for
         ds_idx, ds_y_hat in enumerate(y_hats)]
    ys_denorm = [
        denormalizer(ds_y_hat.detach().cpu(), topic_dataset_packages[ds_idx][3], topic_dataset_packages[ds_idx][4],
                     topic_dataset_packages[ds_idx][5]) for
        ds_idx, ds_y_hat in enumerate(ys)]
    y_hats_denorm = np.stack(y_hats_denorm, axis=0)
    ys_denorm = np.stack(ys_denorm, axis=0)

    # save forecasting values
    save_fsct_y(fcst_val_save_path, 'patent', args.model, '_'.join(args.node_feature_type), ys_denorm, y_hats_denorm)
