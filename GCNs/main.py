import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import torch
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
from _commons.utils import save_metrics, fix_randomness
from config import get_config
from models import get_model, training, evaluating
from preprocessed.preprocessor import get_dataset

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    tb_train_loss = SummaryWriter(log_dir=f'../_tensorboard/{args.node_feature_type}/{args.model}/loss')

    # fix randomness
    fix_randomness(args.seed)

    # load data
    refine_data = True if args.model == 'dcrnn' else False

    # TODO WARNING - check the location of cluster dirs
    cluster_dirs = [os.path.join(args.data_dir, i) for i in os.listdir(args.data_dir)][:100]
    dataset_packages = []
    for _c_dir in cluster_dirs:
        dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
            = get_dataset(_c_dir, args.node_feature_type, args.discard_index, refine_data)
        # FIXME dataloader, validation dataset
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
        dataset_packages.append([train_dataset, test_dataset,
                                 num_nodes, num_features, min_val_tar, max_val_tar, eps])

    # FIXME data parallel : multi-gpu
    # create GCN model
    model = get_model(args, num_nodes, num_features)
    # device_ids = args.device_ids
    # gpu = 'cuda:' + str(args.device_ids[0])
    # model = DataParallel(model, device_ids=device_ids)
    gpu = 'cuda:' + args.device
    device = torch.device(gpu)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss(reduction='mean')

    mse_log = {}
    for epoch in tqdm(range(args.epochs)):

        train_y_hats, train_ys, test_y_hats, test_ys = torch.Tensor().to(device), torch.Tensor().to(
            device), torch.Tensor().to(device), torch.Tensor().to(device)
        for _ds_idx, _curr_dataset_pack in enumerate(dataset_packages):
            # training
            model.train()
            train_dataset = _curr_dataset_pack[0]
            train_y_hat, train_y = training(model, train_dataset, optimizer, criterion, num_features, num_nodes,
                                            args.embedd_dim)

            train_y_hats = torch.concat([train_y_hats, train_y_hat])
            train_ys = torch.concat([train_ys, train_y])

            # evaluating
            model.eval()
            test_dataset = _curr_dataset_pack[1]
            test_y_hat, test_y = evaluating(model, test_dataset, num_features, num_nodes, args.embedd_dim)

            test_y_hats = torch.concat([test_y_hats, test_y_hat])
            test_ys = torch.concat([test_ys, test_y])

        MSE = torch.nn.MSELoss(reduction='mean')
        MAE = torch.nn.L1Loss(reduction='mean')

        train_mse, train_mae = MSE(train_y_hats, train_ys), MAE(train_y_hats, train_ys)
        test_mse, test_mae = MSE(test_y_hats, test_ys), MAE(test_y_hats, test_ys)

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
