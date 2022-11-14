import json
import os
import pathlib

import numpy as np


def get_node_targets(media, topic_num):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_targets = os.path.join(loc_here / media / '4.topic' / str(topic_num), 'word_count.node_targets.npy')
    if os.path.exists(loc_node_targets):
        return np.load(loc_node_targets)

    raise Exception(f'There is no node targets! (media: {media}, topic num: {topic_num})')


def get_edge_indices(media, topic_num):
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
                        if edge_index.size != 0:
                            src_index = np.split(edge_index[0], 2)[0]
                            dst_index = np.split(edge_index[1], 2)[0]
                            edge_index = np.vstack([src_index, dst_index])
                        edge_indices.append(edge_index)

                print('edge indices: {} * {}'.format(len(edge_indices), edge_indices[0].shape))

            return edge_indices

    raise Exception(f'There is no edge indices! (media: {media}, topic num: {topic_num})')


def get_edge_weights(media, topic_num):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_edge_weights = os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_weigths.json')
    if os.path.exists(loc_edge_weights):
        with open(os.path.join(loc_here / media / '4.topic' / str(topic_num), 'cooccurrence.edge_weigths.json'),
                  'r') as f:
            dict_edge_weights = json.load(f)
            with open(f'./preprocessed/{media}/4.topic/{str(topic_num)}/cooccurrence.edge_attributes.txt',
                      'r') as f_eattr:
                edge_attrs = f_eattr.read().splitlines()
                edge_weights = []
                for edge_attr in edge_attrs:
                    if edge_attr in dict_edge_weights:
                        edge_weight = np.array(dict_edge_weights[edge_attr])
                        if edge_weight.size != 0:
                            edge_weight = np.split(edge_weight, 2)[0]
                        edge_weights.append(edge_weight)

                print('edge weights: {} * {}'.format(len(edge_weights), edge_weights[0].shape))

            return edge_weights

    raise Exception(f'There is no edge weights! (media: {media}, topic num: {topic_num})')


def get_node_features(media, topic_num):
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_features = os.path.join(loc_here / media / '4.topic' / str(topic_num),
                                     'betweenness_centrality.inv_cooccurrence.node_targets.npy')
    if os.path.exists(loc_node_features):
        node_features = np.load(loc_node_features)
        node_features = np.expand_dims(node_features, axis=2)
        # FIXME
        node_features = np.nan_to_num(node_features)
        return node_features

    raise Exception(f'There is no node features! (media: {media}, topic num: {topic_num})')
