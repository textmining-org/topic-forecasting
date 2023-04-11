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


def save_metrics(save_path, file_name, args, arg_names, metrics, metric_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    columns = ['timestamp']
    values = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    columns.extend(metric_names)
    values.extend(metrics)
    for arg_name in arg_names:
        arg_value = vars(args).get(arg_name)
        columns.append(arg_name)
        values.append(str(arg_value))

    save_path = os.path.join(save_path, file_name)
    if os.path.exists(save_path):
        df_results = pd.read_csv(save_path)
    else:
        df_results = pd.DataFrame(columns=columns)

    df_results.loc[len(df_results)] = values

    df_results.sort_values(by="mse", ascending=True, inplace=True)
    # print(df_results)
    df_results.to_csv(save_path, index=False)


def exists_metrics(save_path, file_name, args, arg_names):
    if not os.path.exists(os.path.join(save_path, file_name)):
        return False

    df_results = pd.read_csv(os.path.join(save_path, file_name))

    for index, result in df_results.iterrows():
        existence_flag = True
        for arg_name in arg_names:
            if result[arg_name] != vars(args).get(arg_name):
                existence_flag = False
                break

        if existence_flag == True:
            break

    return existence_flag


def save_fsct_y(save_path, media, model_name, node_feature_type, num_training_clusters, true_y, fcst_y):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, f"{media}_{model_name}_{node_feature_type}_{num_training_clusters}_true.npy"), true_y)
    np.save(os.path.join(save_path, f"{media}_{model_name}_{node_feature_type}_{num_training_clusters}_fcst.npy"), fcst_y)


def load_fcst_y(save_path, media, model_name, node_feature_type,  num_training_clusters):
    true_y = np.load(os.path.join(save_path, f"{media}_{model_name}_{node_feature_type}_{num_training_clusters}_true.npy"))
    fcst_y = np.load(os.path.join(save_path, f"{media}_{model_name}_{node_feature_type}_{num_training_clusters}_fcst.npy"))

    return true_y, fcst_y

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
