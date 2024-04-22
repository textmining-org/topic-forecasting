import json
import os
import pathlib

import numpy as np
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, DynamicGraphTemporalSignalBatch
import torch
from typing import List, Union
from torch_geometric.data import Data


def get_node_targets(data_path, discard_index=None):
    loc_node_targets = os.path.join(data_path, 'word_count.node_targets.npy')
    if os.path.exists(loc_node_targets):
        node_targets = np.load(loc_node_targets)
        if discard_index is not None:
            node_targets = node_targets[discard_index:]

        # print('node targets: {}'.format(node_targets.shape))
        return node_targets

    raise Exception(f'There is no node targets! (data_path: {data_path})')


def get_edge_indices(data_path, bidirectional=False, discard_index=None):
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
                        if bidirectional == True and edge_index.size != 0:
                            src_index = np.split(edge_index[0], 2)[0]
                            dst_index = np.split(edge_index[1], 2)[0]
                            edge_index = np.vstack([src_index, dst_index])
                        edge_indices.append(edge_index)

            if discard_index is not None:
                edge_indices = edge_indices[discard_index:]

            # print(f'edge indices: {len(edge_indices)} * {edge_indices[0].shape}')
            return edge_indices

    raise Exception(f'There is no edge indices! (data_path: {data_path})')


def get_edge_weights(data_path, bidirectional=False, discard_index=None):
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

            if discard_index is not None:
                edge_weights = edge_weights[discard_index:]

            # print('edge weights: {} * {}'.format(len(edge_weights), edge_weights[0].shape))
            return edge_weights

    raise Exception(f'There is no edge weights! (data_path: {data_path})')


def get_node_features(data_path, feature_type=['betweenness'], discard_index=None):
    node_features = None
    for type in feature_type:
        loc_node_feature = os.path.join(data_path, f'{type}_centrality.inv_cooccurrence.node_targets.npy')
        if os.path.exists(loc_node_feature):
            node_feature = np.load(loc_node_feature)
            node_feature = np.expand_dims(node_feature, axis=2)
            # FIXME
            node_feature = np.nan_to_num(node_feature)

            if discard_index is not None:
                node_feature = node_feature[discard_index:]

            if node_features is None:
                node_features = node_feature
            else:
                node_features = np.concatenate([node_features, node_feature], axis=2)

    # print('node features: {}'.format(node_features.shape))
    return node_features


def refine_graph_data(node_targets, node_features, edge_indices, edge_weights):
    """
    node targets: (64, 30)
    node feature: (64, 30, 1)
    edge indices: 64 * (2, 0)
    edge weights: 64 * (0,)

    """
    print(f'node targets: {node_targets.shape}')
    print(f'node feature: {node_features.shape}')
    print(f'edge indices: {len(edge_indices)} * {edge_indices[0].shape}')
    print(f'edge weights: {len(edge_weights)} * {edge_weights[0].shape}')

    refined_node_targets = []
    refined_node_features = []
    refined_edge_indices = []
    refined_edge_weights = []

    time_seq_len = node_targets.shape[0]
    for i in range(time_seq_len):
        if edge_indices[i].any() == True:
            refined_node_targets.append(node_targets[i])
            refined_node_features.append(node_features[i])
            refined_edge_indices.append(edge_indices[i])
            refined_edge_weights.append(edge_weights[i])

    refined_node_targets = np.array(refined_node_targets)
    refined_node_features = np.array(refined_node_features)
    print(f'refined node targets: {refined_node_targets.shape}')
    print(f'refined node feature: {refined_node_features.shape}')
    print(f'refined edge indices: {len(refined_edge_indices)} * {refined_edge_indices[0].shape}')
    print(f'refined edge weights: {len(refined_edge_weights)} * {refined_edge_weights[0].shape}')

    return refined_node_targets, refined_node_features, refined_edge_indices, refined_edge_weights


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


class CDGTS(DynamicGraphTemporalSignal):
    def __getitem__(self, time_index: Union[int, slice]):
        if isinstance(time_index, slice):
            snapshot = DynamicGraphTemporalSignal(
                self.edge_indices[time_index],
                self.edge_weights[time_index],
                self.features[time_index],
                self.targets[time_index],
                **{key: getattr(self, key)[time_index] for key in self.additional_feature_keys}
            )
        else:
            x = self._get_features(time_index)
            edge_index = self._get_edge_index(time_index)
            edge_weight = self._get_edge_weight(time_index)
            # FIXME to forecast next timestep
            y = self._get_target(time_index + 1)
            additional_features = self._get_additional_features(time_index)

            snapshot = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                            y=y, **additional_features)
        return snapshot

    def __next__(self):
        # FIXME to forecast next timestep
        if self.t < len(self.features) - 1:
            snapshot = self[self.t]
            self.t = self.t + 1
            return snapshot
        else:
            self.t = 0
            raise StopIteration


def get_dataset(data_path, node_feature_type=['betweenness'], discard_index=None, refine_data=False):
    # node targets(label)
    node_targets = get_node_targets(data_path, discard_index=discard_index)
    node_targets, min_val_tar, max_val_tar, eps = normalizer(node_targets)
    num_nodes = node_targets[0].shape[0]

    # node features
    node_features = get_node_features(data_path, node_feature_type, discard_index=discard_index)
    node_features, min_val_fea, max_val_fea, eps = normalizer(node_features)
    num_features = node_features[0].shape[1]

    # edge indices and weights
    edge_indices = get_edge_indices(data_path, discard_index=discard_index)
    edge_weights = get_edge_weights(data_path, discard_index=discard_index)

    if refine_data == True:
        node_targets, node_features, edge_indices, edge_weights = refine_graph_data(node_targets, node_features,
                                                                                    edge_indices, edge_weights)



    print(node_features.shape)
    print(node_targets.shape)
    # print(edge_indices)
    # print(edge_indices)
    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, node_features, node_targets)

    return dataset, num_nodes, num_features, min_val_tar, max_val_tar, eps
