import json
import os
from typing import Union
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


# (32, 50, 3) : (time, node, pred_len)
def get_node_targets(data_path, seq_len=1, pred_len=1):
    # print(data_path, seq_len, pred_len)
    loc_node_targets = os.path.join(data_path, 'word_count.node_targets.npy')

    if os.path.exists(loc_node_targets):
        node_targets = np.load(loc_node_targets)
        # print(f'node_targets: {node_targets.shape}, {node_targets}')
        node_targets_seq = []
        # for i in range(node_targets.shape[0] - int(seq_len + pred_len - 1)):
        for i in range(node_targets.shape[0] - int(seq_len + pred_len + 1)):
            seq = node_targets[i:i + seq_len + pred_len, :]
            # print(i, seq.shape)
            node_targets_seq.append(seq[seq_len:seq_len + pred_len, :])

        node_targets_seq = np.array(node_targets_seq)
        # print(node_targets_seq.shape)
        node_targets_seq = np.moveaxis(node_targets_seq, 1, 2)
        return node_targets_seq

    raise Exception(f'There is no node targets! (data_path: {data_path})')


# (32, 50, 4, 12) : (time, node, features, seq_len)
def get_node_features(data_path, feature_type=['betweenness', 'closeness', 'degree'], seq_len=1, pred_len=1):
    loc_node_targets = os.path.join(data_path, 'word_count.node_targets.npy')
    node_targets = np.expand_dims(np.load(loc_node_targets), axis=2)

    node_features = None
    for type in feature_type:
        loc_node_feature = os.path.join(data_path, f'{type}_centrality.inv_cooccurrence.node_targets.npy')
        if os.path.exists(loc_node_feature):
            node_feature = np.load(loc_node_feature)
            node_feature = np.expand_dims(node_feature, axis=2)
            # FIXME
            node_feature = np.nan_to_num(node_feature)

            if node_features is None:
                node_features = node_feature
            else:
                node_features = np.concatenate([node_features, node_feature], axis=2)

    # (48, 50, 4) : (time, node, features)
    node_features = np.concatenate((node_features, node_targets), axis=2)

    node_features_seq = []
    # for i in range(node_features.shape[0] - int(seq_len + pred_len - 1)):
    for i in range(node_features.shape[0] - int(seq_len + pred_len + 1)):
        seq = node_features[i:i + seq_len + pred_len, :, :]
        node_features_seq.append(seq[:seq_len, :, :])

    node_features_seq = np.moveaxis(np.array(node_features_seq), 1, 2)
    node_features_seq = np.moveaxis(node_features_seq, 2, 3)
    return node_features_seq

def get_edge_indices(data_path, bidirectional=False, seq_len=1, pred_len=1):
    loc_edge_indices = os.path.join(data_path, 'cooccurrence.edge_indices.json')
    if os.path.exists(loc_edge_indices):
        with open(os.path.join(data_path, 'cooccurrence.edge_indices.json'), 'r') as f:
            dict_edge_indices = json.load(f)
            with open(os.path.join(data_path, 'cooccurrence.edge_attributes.txt'), 'r') as f_eattr:
                edge_attrs = f_eattr.read().splitlines()
                edge_indices = []
                for edge_attr in edge_attrs:
                    if edge_attr in dict_edge_indices:

                        edge_index = np.array(dict_edge_indices[edge_attr])
                        # print(f'##### edge_attr: {edge_attr}, edge_index: {edge_index}')
                        # 2024.04.22 : 현재 데이터는 무방향으로, 아래의 IF 구문이 실행되지 않음
                        if bidirectional == True and edge_index.size != 0:
                            src_index = np.split(edge_index[0], 2)[0]
                            dst_index = np.split(edge_index[1], 2)[0]
                            edge_index = np.vstack([src_index, dst_index])
                        edge_indices.append(edge_index)

            edge_indices_seq = []
            # for i in range(len(edge_indices) - int(seq_len + pred_len - 1)):
            for i in range(len(edge_indices) - int(seq_len + pred_len + 1)):
                seq = edge_indices[i:i + seq_len + pred_len]
                edge_indices_seq.append(seq[:seq_len])

            return edge_indices_seq

    raise Exception(f'There is no edge indices! (data_path: {data_path})')


def get_edge_weights(data_path, bidirectional=False, seq_len=1, pred_len=1):
    loc_edge_weights = os.path.join(data_path, 'cooccurrence.edge_weights.json')
    if os.path.exists(loc_edge_weights):
        with open(os.path.join(data_path, 'cooccurrence.edge_weights.json'), 'r') as f:
            dict_edge_weights = json.load(f)
            with open(os.path.join(data_path, 'cooccurrence.edge_attributes.txt'), 'r') as f_eattr:
                edge_attrs = f_eattr.read().splitlines()
                edge_weights = []
                for edge_attr in edge_attrs:
                    if edge_attr in dict_edge_weights:
                        edge_weight = np.array(dict_edge_weights[edge_attr])
                        if bidirectional == True and edge_weight.size != 0:
                            edge_weight = np.split(edge_weight, 2)[0]
                        edge_weights.append(edge_weight)

            # print('edge weights: {} * {}'.format(len(edge_weights), edge_weights[0].shape))

            edge_weights_seq = []
            # for i in range(len(edge_weights) - int(seq_len + pred_len - 1)):
            for i in range(len(edge_weights) - int(seq_len + pred_len + 1)):
                seq = edge_weights[i:i + seq_len + pred_len]
                edge_weights_seq.append(seq[:seq_len])

            return edge_weights_seq

    raise Exception(f'There is no edge weights! (data_path: {data_path})')

def normalizer(data):
    """
    Execute min-max normalization to data

    Args:
        data: target data

    Returns:
        norm_data: normalized data

    """
    # norm_data = (data - min) / (max + min + 1e-7)
    min_val = np.min(data, 0)
    max_val = np.max(data, 0)
    eps = 1e-7

    numerator = data - min_val
    denominator = max_val - min_val
    norm_data = numerator / (denominator + eps)

    return norm_data, min_val, max_val, eps


def denormalizer(data, min_val, max_val, eps):
    denorm_data = data * (max_val - min_val + eps) + min_val
    return denorm_data

def get_loader(data_path, node_feature_type=['betweenness', 'closeness', 'degree'], seq_len=1, pred_len=1, batch_size=4,
               device='cpu'):
    # node targets(label)
    node_targets = get_node_targets(data_path, seq_len, pred_len)
    node_targets, min_val_tar, max_val_tar, eps = normalizer(node_targets)
    num_nodes = node_targets[0].shape[0]

    # node features
    node_features = get_node_features(data_path, node_feature_type, seq_len, pred_len)
    node_features, min_val_fea, max_val_fea, eps = normalizer(node_features)
    num_features = node_features[0].shape[1]
    min_val_fea = np.squeeze(min_val_fea[:, -1:, :])
    max_val_fea = np.squeeze(max_val_fea[:, -1:, :])

    # edge indices and weights
    edge_indices = get_edge_indices(data_path, seq_len=seq_len, pred_len=pred_len)
    edge_weights = get_edge_weights(data_path, seq_len=seq_len, pred_len=pred_len)

    node_feature_tensor = torch.from_numpy(node_features.astype('float32')).to(device)
    node_target_tensor = torch.from_numpy(node_targets.astype('float32')).to(device)
    # node_feature_tensor = torch.from_numpy(node_features).type(torch.FloatTensor).to(device)
    # node_target_tensor = torch.from_numpy(node_targets).type(torch.FloatTensor).to(device)
    dataset = TensorDataset(node_feature_tensor, node_target_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

    edge_indices_batch = []
    edge_weights_batch = []
    for idx, value in enumerate(dataloader):
        edge_batch_start_idx = idx * batch_size
        edge_batch_end_idx = len(edge_indices) \
            if ((idx + 1) * batch_size) > len(edge_indices) else (idx + 1) * batch_size
        edge_indices_batch.append(edge_indices[edge_batch_start_idx:edge_batch_end_idx])
        edge_weights_batch.append(edge_weights[edge_batch_start_idx:edge_batch_end_idx])

    return dataloader, edge_indices_batch, edge_weights_batch, num_nodes, num_features, min_val_fea, max_val_fea, min_val_tar, max_val_tar, eps


# data_path = '/Data2/yejin/blockchain_data/papers_co10/clusters.max_structured.time_split/train/random_cluster_4000'
# data_path = '/Data2/yejin/blockchain_data/papers_co10/clusters.max_structured.time_split/train/random_cluster_0000'
# batch_size = 4
# seq_len = 12
# dataloader, edge_indices_batch, edge_weights_batch, num_nodes, num_features, min_val_tar, max_val_tar, eps = get_loader(
#     data_path,
#     seq_len=seq_len,
#     pred_len=3,
#     batch_size=batch_size)

# [batch_num, [
# print(len(edge_indices_batch))
#
# edge_indices_batch_rep_list = []
# for i in range(len(dataloader)):
#     edge_indices_batch_rep = None
#     for j in range(batch_size):
#         for k in range(seq_len):
#             edge_indices = edge_indices_batch[i][j][k]
#             # print(edge_indices)
#
#             if edge_indices.shape[1] == 0:
#                 continue
#
#             if edge_indices_batch_rep is None:
#                 edge_indices_batch_rep = edge_indices
#             else:
#                 edge_indices_batch_rep = np.concatenate((edge_indices_batch_rep, edge_indices), axis=1)
#     print(f'edge_indices_batch_rep: {edge_indices_batch_rep}')
#     edge_indices_batch_rep_list.append(edge_indices_batch_rep)
#
#
# for edge_indices_batch_rep in edge_indices_batch_rep_list:
#     print(f'edge_indices_batch_rep: {edge_indices_batch_rep}')
#     # 중복 제거
#     adj_matrix = np.zeros((edge_indices_batch_rep.max() + 1, edge_indices_batch_rep.max() + 1))
#     adj_matrix[edge_indices_batch_rep[0, :], edge_indices_batch_rep[1, :]] = 1
#     adj_matrix[edge_indices_batch_rep[1, :], edge_indices_batch_rep[0, :]] = 1
#
#     edges = np.nonzero(adj_matrix)
#     print(edges)
#     print(np.array(edges))

