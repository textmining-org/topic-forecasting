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
import math

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    model_save_path = os.path.join(results_path, 'models')

    tb_train_loss = SummaryWriter(log_dir=f'../_tensorboard/{args.node_feature_type}/{args.model}/loss')

    # fix randomness
    fix_randomness(args.seed)

    # load data
    refine_data = True if args.model == 'dcrnn' else False

    # TODO WARNING - check the location of cluster dirs
    cluster_dirs = [os.path.join(args.cluster_dir, i) for i in os.listdir(args.cluster_dir)][:100]
    dataset_packages = []
    for _c_dir in cluster_dirs:
        dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
            = get_dataset(_c_dir, args.node_feature_type, args.discard_index, refine_data)
        # FIXME dataloader, validation dataset
        train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.7)
        valid_dataset, test_dataset = temporal_signal_split(test_dataset, train_ratio=0.5)
        dataset_packages.append([train_dataset, valid_dataset, test_dataset,
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

    # mse_log = {}
    best_epoch, best_train_mse, best_valid_mse, best_test_mse = 0, math.inf, math.inf, math.inf
    for epoch in tqdm(range(args.epochs)):

        train_y_hats, train_ys, valid_y_hats, valid_ys, test_y_hats, test_ys \
            = torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(device), torch.Tensor().to(
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
            valid_dataset = _curr_dataset_pack[1]
            valid_y_hat, valid_y = evaluating(model, valid_dataset, num_features, num_nodes, args.embedd_dim)

            valid_y_hats = torch.concat([valid_y_hats, valid_y_hat])
            valid_ys = torch.concat([valid_ys, valid_y])

            test_dataset = _curr_dataset_pack[2]
            test_y_hat, test_y = evaluating(model, test_dataset, num_features, num_nodes, args.embedd_dim)

            test_y_hats = torch.concat([test_y_hats, test_y_hat])
            test_ys = torch.concat([test_ys, test_y])

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
            torch.save(model.state_dict(), os.path.join(model_save_path, f'{args.model}_{args.node_feature_type}.pt'))

    print("[Final (BEST MSE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
        best_train_mse, best_valid_mse, best_test_mse, best_epoch))

    print("[Final (BEST MAE)] Train: {:.8f} | Valid : {:.8f} | Test : {:.8f} at Epoch {:3}".format(
        best_train_mae, best_valid_mae, best_test_mae, best_epoch, best_epoch))


    # topic evaluation
    model.load_state_dict(torch.load(os.path.join(model_save_path, f'{args.model}_{args.node_feature_type}.pt')))
    model.eval()

    topic_dirs = [os.path.join(args.topic_dir, i) for i in os.listdir(args.topic_dir)]
    topic_dataset_packages = []
    for _t_dir in topic_dirs:
        print(_t_dir)
        topic_dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps \
            = get_dataset(_t_dir, args.node_feature_type, args.discard_index, refine_data)
        topic_dataset_packages.append([topic_dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps])

    topic_y_hats, topic_ys = torch.Tensor().to(device), torch.Tensor().to(device)
    for _ds_idx, _curr_dataset_pack in enumerate(topic_dataset_packages):
        dataset = _curr_dataset_pack[0]
        topic_y_hat, topic_y = evaluating(model, dataset, num_features, num_nodes, args.embedd_dim)

        topic_y_hats = torch.concat([topic_y_hats, topic_y_hat])
        topic_ys = torch.concat([topic_ys, topic_y])

    MSE = torch.nn.MSELoss(reduction='mean')
    MAE = torch.nn.L1Loss(reduction='mean')
    topic_mse, topic_mae = MSE(topic_y_hats, topic_ys).item(), MAE(topic_y_hats, topic_ys).item()

    # save cluster metrics
    arg_names = ['model', 'node_feature_type', 'epochs', 'lr', 'cluster_dir', 'topic_dir']
    metric_names = ['mse', 'mae', 'topic_mse', 'topic_mae']
    metrics = [best_test_mse, best_test_mae, topic_mse, topic_mae]
    save_metrics(results_path, 'metrics.csv', args, arg_names, metrics, metric_names)


    # FIXME bug : out of index
    # de-normalizing
    # denorm_y_hats = [denormalizer(ds_y_hat, dataset_packages[ds_idx][4], dataset_packages[ds_idx][5], dataset_packages[ds_idx][6]) for
    #                  ds_idx, ds_y_hat in enumerate(test_y_hats)]
    # denorm_ys = [denormalizer(ds_y_hat, dataset_packages[ds_idx][4], dataset_packages[ds_idx][5], dataset_packages[ds_idx][6]) for
    #              ds_idx, ds_y_hat in enumerate(test_ys)]
    #
    # save_pred_y(results_path, args.data_dir, args.topic_num, args.model, denorm_ys, denorm_y_hats)
