import json
import os
import pathlib

import numpy as np


def get_node_targets(media, topic_num, discard_index=None):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_targets = os.path.join(loc_here / media / '4.topic' / str(topic_num), 'word_count.node_targets.npy')
    if os.path.exists(loc_node_targets):
        node_targets = np.load(loc_node_targets)
        if discard_index is not None:
            node_targets = node_targets[discard_index:]

        print('node targets: {}'.format(node_targets.shape))
        return node_targets

    raise Exception(f'There is no node targets! (media: {media}, topic num: {topic_num})')


def get_edge_indices(media, topic_num, bidirectional=False, discard_index=None):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_edge_indices = os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_indices.json')
    if os.path.exists(loc_edge_indices):
        with open(os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_indices.json'),
                  'r') as f:
            dict_edge_indices = json.load(f)
            with open(f'./preprocessed/{media}/4.topic/{str(topic_num)}/cooccurrence.edge_attributes.txt',
                      'r') as f_eattr:
                edge_attrs = f_eattr.read().splitlines()
                edge_indices = []
                for edge_attr in edge_attrs:
                    if edge_attr in dict_edge_indices:
                        edge_index = np.array(dict_edge_indices[edge_attr])
                        if bidirectional == True and edge_index.size != 0:
                            print(edge_index.shape)
                            src_index = np.split(edge_index[0], 2)[0]
                            dst_index = np.split(edge_index[1], 2)[0]
                            edge_index = np.vstack([src_index, dst_index])
                        edge_indices.append(edge_index)

            if discard_index is not None:
                edge_indices = edge_indices[discard_index:]

            print(f'edge indices: {len(edge_indices)} * {edge_indices[0].shape}')
            return edge_indices

    raise Exception(f'There is no edge indices! (media: {media}, topic num: {topic_num})')


def get_edge_weights(media, topic_num, bidirectional=False, discard_index=None):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_edge_weights = os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_weights.json')
    if os.path.exists(loc_edge_weights):
        with open(os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_weights.json'),
                  'r') as f:
            dict_edge_weights = json.load(f)
            with open(f'./preprocessed/{media}/4.topic/{str(topic_num)}/cooccurrence.edge_attributes.txt',
                      'r') as f_eattr:
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

            print('edge weights: {} * {}'.format(len(edge_weights), edge_weights[0].shape))
            return edge_weights

    raise Exception(f'There is no edge weights! (media: {media}, topic num: {topic_num})')


def get_node_features(media, topic_num, discard_index=None):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_features = os.path.join(loc_here / media / '4.topic' / str(topic_num),
                                     'betweenness_centrality.inv_cooccurrence.node_targets.npy')
    if os.path.exists(loc_node_features):
        node_features = np.load(loc_node_features)
        node_features = np.expand_dims(node_features, axis=2)
        # FIXME
        node_features = np.nan_to_num(node_features)

        if discard_index is not None:
            node_features = node_features[discard_index:]

        print('node feature: {}'.format(node_features.shape))
        return node_features

    raise Exception(f'There is no node features! (media: {media}, topic num: {topic_num})')

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

    refined_node_targets  = []
    refined_node_features = []
    refined_edge_indices = []
    refined_edge_weights = []

    del_indices = []
    print(node_targets.shape)
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

    numerator = data - min_val
    denominator = max_val - min_val
    norm_data = numerator / (denominator + 1e-7)

    return norm_data, min_val, max_val
