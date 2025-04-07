import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)


def save_metrics(save_path, args, arg_names, metrics, metric_names):
    columns = ['timestamp']
    values = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    columns.extend(metric_names)
    values.extend(metrics)
    for arg_name in arg_names:
        arg_value = vars(args).get(arg_name)
        columns.append(arg_name)
        values.append(str(arg_value))

    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=columns)

    df_results.loc[len(df_results)] = values

    df_results.sort_values(by="mse", ascending=True, inplace=True)
    print(df_results)
    df_results.to_csv(save_path, index=False)


def exists_metrics(save_path, args, arg_names):
    if not os.path.exists(save_path):
        return False

    df_results = pd.read_csv(save_path)

    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            result_item = result[arg_name]
            args_item = vars(args).get(arg_name)

            if type(args_item) is list:
                result_item = result_item.replace('[', '').replace(']', '').replace('\'', '').replace('\'', '').replace(
                    ' ', '').split(sep=',')

            if result_item != args_item:
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag


def save_fsct_y(save_path, args, true_y, fcst_y, x=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    node_feature_type = '_'.join(args.node_feature_type)

    file_name=None
    if args.model == 'a3tgcn2':
        file_name = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.out_channels}'
    elif args.model == 'agcrn':
        file_name = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.K}_{args.embedd_dim}_{args.out_channels}'
    elif args.model in ['lstm', 'gru']:
        file_name = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.hidden_size}_{args.num_layers}'

    np.save(os.path.join(save_path, f"{file_name}_true.npy"), true_y)
    np.save(os.path.join(save_path, f"{file_name}_fcst.npy"), fcst_y)

    if x is not None:
        np.save(os.path.join(save_path, f"{file_name}_x.npy"), x)


def load_fcst_y(save_path, args, model, load_x=False):
    node_feature_type = '_'.join(args.best_node_feature_type)
    file_name=None
    if model == 'a3tgcn2':
        file_name = f'{args.media}_{model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.best_a3tgcn2_out_channels}'
    elif model == 'agcrn':
        file_name = f'{args.media}_{model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.best_agcrn_K}_{args.best_agcrn_embedd_dim}_{args.best_agcrn_out_channels}'
    elif model == 'lstm':
        file_name = f'{args.media}_{model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.best_lstm_hidden_size}_{args.best_lstm_num_layers}'
    elif model == 'gru':
        file_name = f'{args.media}_{model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{args.best_gru_hidden_size}_{args.best_gru_num_layers}'

    true_y = np.load(os.path.join(save_path, f"{file_name}_true.npy"))
    fcst_y = np.load(os.path.join(save_path, f"{file_name}_fcst.npy"))
    if load_x == False:
        return true_y, fcst_y
    else:
        x = np.load(os.path.join(save_path, f"{file_name}_x.npy"))
        return true_y, fcst_y, x


def save_fcst_topic_order(save_path, file_name, media, topic_dirs):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    columns = ['media', 'topic_dir', 'topic_order']
    save_path = os.path.join(save_path, file_name)
    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path, sep='\t')
    else:
        df_results = pd.DataFrame(columns=columns)

    for i, topic_dir in enumerate(topic_dirs):
        df_results.loc[len(df_results)] = [media, topic_dir, i]

    df_results.to_csv(save_path, index=False, sep='\t')


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, valid_loss):
        score = -valid_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'##### EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def adjust_learning_rate(optimizer, epoch, lr, type):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if type == '1':
        lr_adjust = {epoch: lr * (0.5 ** ((epoch - 1) // 1))}
    elif type == '2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif type == '3':
        lr_adjust = {epoch: lr if epoch < 10 else lr*0.1}
    elif type == '4':
        lr_adjust = {epoch: lr if epoch < 15 else lr*0.1}
    elif type == '5':
        lr_adjust = {epoch: lr if epoch < 25 else lr*0.1}
    elif type == '6':
        lr_adjust = {epoch: lr if epoch < 5 else lr*0.1}
    elif type == 'constant':
        lr_adjust = {epoch: lr}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

