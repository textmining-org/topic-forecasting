import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

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
    node_feature_type = '_'.join(args.node_feature_type)
    model_filename = f'{args.model}_{node_feature_type}_{args.num_training_clusters}.pt'

    # load data
    # FIXME error : DCRNN에서 refine 수행 시 shape 불일치 오류 발생 (yhat-(55, 50), y-(57, 50)), 임시로 False 처리
    #
    # Traceback (most recent call last):
    #   File "main_fcst.py", line 50, in <module>
    #     y_hats = torch.concat([y_hats, y_hat[None, :, :]])
    # RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 55 but got size 57 for tensor number 1 in the list.
    # refine_data = True if args.model == 'dcrnn' else False
    refine_data = False
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
    print(model)

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
    arg_names = ['model', 'node_feature_type', 'num_training_clusters', 'topic_dir']
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
    save_fsct_y(fcst_val_save_path, 'patent', args.model, node_feature_type, args.num_training_clusters, ys_denorm, y_hats_denorm)
