import os
import sys

parent_path = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(parent_path)

import numpy as np
import pandas as pd
import logging
import json
import pathlib

from network_analysis.graph_reconstruction import load_graph
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal, StaticGraphTemporalSignal


def get_node_indices():
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_indices = os.path.join(loc_here, 'node_indices.tsv')
    if os.path.exists(loc_node_indices):
        return pd.read_csv(loc_node_indices, sep='\t', names=['node_index', 'node_name'])

    loc_graph = os.path.join(loc_here, 'combined_graph.pkl')
    assert os.path.exists(loc_graph), "There does not exist graph file(named 'combined graph.pkl)"

    G = load_graph(loc_graph)
    df_node_indices = pd.DataFrame({'node_index': '', 'node_name': G.nodes})
    df_node_indices['node_index'] = df_node_indices.index
    df_node_indices.to_csv(loc_node_indices, sep='\t', header=False, index=False)

    return df_node_indices


def get_node_targets():
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_node_targets = os.path.join(loc_here, 'node_targets.npy')
    if os.path.exists(loc_node_targets):
        return np.load(loc_node_targets)

    loc_graph = os.path.join(loc_here, 'combined_graph.pkl')
    assert os.path.exists(loc_graph), "There does not exist graph file(named 'combined graph.pkl)"

    loc_node_indices = os.path.join(loc_here, 'node_indices.tsv')
    assert os.path.exists(loc_node_indices), "There does not exist node indices file(named 'node_indices.tsv')"

    loc_node_attr = os.path.join(loc_here, 'node_attributes.txt')
    assert os.path.exists(loc_node_attr), "There does not exist node attributes file(named 'node_attributes.txt')"

    G = load_graph(loc_graph)
    node_indices = pd.read_csv(loc_node_indices, sep='\t', names=['node_index', 'node_name'])
    node_names = node_indices['node_name'].tolist()

    with open(loc_node_attr, 'r') as f_nattr:
        node_attrs = f_nattr.read().splitlines()
        targets = []

        for node_attr in node_attrs:
            target = []
            for node_name in node_names:
                node_attr = node_attr.strip()
                target.append(G.nodes[node_name][node_attr])
            targets.append(np.asarray(target))
        f_nattr.close()
        np.save(loc_node_targets, targets)

    return np.load(loc_node_targets)


def get_edge_list():
    loc_here = pathlib.Path(__file__).resolve().parent

    loc_edge_list = os.path.join(loc_here, 'edge_list.tsv')
    if os.path.exists(loc_edge_list):
        return pd.read_csv('edge_list.tsv', sep='\t', names=['u', 'v'])

    loc_graph = os.path.join(loc_here, 'combined_graph.pkl')
    assert os.path.exists(loc_graph), "There does not exist graph file(named 'combined graph.pkl)"
    G = load_graph(loc_graph)

    df_edges = pd.DataFrame(G.edges, columns=['u', 'v'])
    df_edges.to_csv(loc_edge_list, sep='\t', header=False, index=False)

    return df_edges


def get_edge_indices_and_weights():
    # logging.basicConfig(filename='get_edge_weights.log', level=logging.DEBUG)

    loc_here = pathlib.Path(__file__).resolve().parent

    loc_edge_indices = os.path.join(loc_here, 'edge_indices.json')
    loc_edge_weights = os.path.join(loc_here, 'edge_weights.json')
    if os.path.exists(loc_edge_indices) and os.path.exists(loc_edge_weights):
        with open(os.path.join(loc_here, 'edge_indices.json'), 'r') as f_i, \
                open(os.path.join(loc_here, 'edge_weights.json'), 'r') as f_w:
            edge_indices = json.load(f_i)
            edge_weights = json.load(f_w)

            return edge_indices, edge_weights

    loc_graph = os.path.join(loc_here, 'combined_graph.pkl')
    assert os.path.exists(loc_graph), "There does not exist graph file(named 'combined graph.pkl)"

    loc_node_indices = os.path.join(loc_here, 'node_indices.tsv')
    assert os.path.exists(loc_node_indices), "There does not exist node indices file(named 'node_indices.tsv')"

    loc_edge_list = os.path.join(loc_here, 'edge_list.tsv')
    assert os.path.exists(loc_edge_list), "There does not exist edge list file(named 'edge_list.tsv')"

    loc_edge_attr = os.path.join(loc_here, 'edge_attributes.txt')
    assert os.path.exists(loc_edge_attr), "There does not exist edge attributes file(named 'edge_attributes.txt')"

    G = load_graph(loc_graph)
    node_indices = pd.read_csv(loc_node_indices, sep='\t', names=['node_index', 'node_name'])
    edge_list = pd.read_csv(loc_edge_list, sep='\t', names=['u', 'v'])

    with open(loc_edge_attr, 'r') as f_eattr:
        edge_attrs = f_eattr.read().splitlines()
        dict_edge_indices = {}
        dict_edge_weights = {}

        for edge_attr in edge_attrs:
            node_u_indices = []
            node_v_indices = []
            edge_weights = []

            for i, edge in edge_list.iterrows():
                edge_data = G.get_edge_data(u=edge['u'], v=edge['v'], default=0)
                edge_weight = edge_data.get(edge_attr, 0)

                if i % 10000 == 0:
                    # logging.debug('{} {} {} {}'.format(i, edge.values, edge_attr, edge_weight))
                    print('{} {} {} {}'.format(i, edge.values, edge_attr, edge_weight))

                if edge_weight != 0:
                    node_u_index = node_indices[node_indices['node_name'] == edge['u']]['node_index'].values[0]
                    node_v_index = node_indices[node_indices['node_name'] == edge['v']]['node_index'].values[0]
                    node_u_indices.append(node_u_index)
                    node_v_indices.append(node_v_index)

                    edge_weights.append(edge_weight)

            if len(node_u_indices) != 0 and len(node_v_indices) != 0:
                dict_edge_indices[edge_attr] = np.stack((node_u_indices, node_v_indices), axis=0).tolist()

            if len(edge_weights) != 0:
                dict_edge_weights[edge_attr] = edge_weights

        # FIXME return JSON → return numpy
        # edge_indices = []
        # edge_weights = []
        # for edge_attr in edge_attrs:
        #     edge_indices.append(np.array(dict_edge_indices[edge_attr]))
        #     edge_weights.append(np.array(dict_edge_weights[edge_attr]))
        #
        # np.save(loc_edge_indices, edge_indices)
        # np.save(loc_edge_weights, edge_weights)

        with open(loc_edge_indices, 'w') as f_i, open(loc_edge_weights, 'w') as f_w:
            json.dump(dict_edge_indices, f_i)
            json.dump(dict_edge_weights, f_w)

    return dict_edge_indices, dict_edge_indices

if __name__ == "__main__":

    get_node_indices()
    get_node_targets()
    get_edge_list()

    dict_edge_indices, dict_edge_weights = get_edge_indices_and_weights()
    output_dir = os.path.abspath('./')
    with open(os.path.join(output_dir, 'edge_attributes.txt'), 'r') as f_eattr:
        edge_attrs = f_eattr.read().splitlines()

        edge_indices = []
        edge_weights = []
        for edge_attr in edge_attrs:
            if edge_attr in dict_edge_indices and edge_attr in dict_edge_weights:
                edge_indices.append(np.array(dict_edge_indices[edge_attr]))
                edge_weights.append(np.array(dict_edge_weights[edge_attr]))

        print('edge indices length: {}'.format(len(edge_indices)))
        print('edge indices[0] shape: {}'.format(edge_indices[1].shape))

        print('edge weights length: {}'.format(len(edge_weights)))
        print('edge weights[0] shape: {}'.format(edge_weights[1].shape))
