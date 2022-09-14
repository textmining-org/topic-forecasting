import os

import numpy as np
import pandas as pd

from network_analysis.coword_detection import parse_preprocessed_data
from network_analysis.graph_reconstruction import load_graph
from network_analysis.graph_reconstruction import reconstruct_graph
import json

# import graph_analysis

def test_make_graph():
    input_f = os.path.abspath('../_datasets/pre_patents.pkl')
    output_dir = os.path.abspath('./')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Parsing preprocessed data... \n{input_f}')
    parse_preprocessed_data(
        data_file=input_f,
        output_dir=output_dir,
    )

    _coword_chunk_file_ = os.path.join(output_dir, 'coword_results.pkl')
    print(f'Data parsing has been finished :\n{_coword_chunk_file_}')

    print(f'Reconstructing master graph... \n{_coword_chunk_file_}')
    G, node_annotations, edge_annotations = reconstruct_graph(
        coword_file=_coword_chunk_file_,
        word_count_annotation_prefix='word_count',  # annotation prefix for node
        whole_word_count_annotation='word_count:whole_time',  # annotation of node for whole time
        coword_annotation_prefix='cooccurrence',  # annotation prefix for edge
        whole_coword_annotation='cooccurrence:whole_time',  # annotation of edge for whole time
        output_dir=output_dir,
    )

    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    node_annotation_file = os.path.join(output_dir, 'node_attributes.txt')
    edge_annotation_file = os.path.join(output_dir, 'edge_attributes.txt')
    print(f'Master graph reconstruction has been finished :\n{_coword_chunk_file_}')


def test_graph():
    output_dir = os.path.abspath('./')
    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    # # \Lib\site-packages\networkx\classes\graph.py
    G = load_graph(graph_file)

    with open(os.path.join(output_dir, 'edge_indices.txt'), 'r') as f_eindices, \
            open(os.path.join(output_dir, 'edge_attributes.txt'), 'r') as f_eattr:
        edge_indices = [tuple(map(str, edge_index.strip().split(' '))) for edge_index in f_eindices]
        edge_attrs = f_eattr.readlines()

        edge_weights = []
        for edge_index in edge_indices:
            print(edge_index)
            edge_weight = []
            for edge_attr in edge_attrs:
                edge_attr = edge_attr.strip()
                edge_weight.append(G[edge_index[0]][edge_index[1]][edge_attr])
            edge_weights.append(np.asarray(edge_weight))
        f_eindices.close()
        f_eattr.close()
        np.save('./edge_weights.npy', edge_weights)

    edge_weights = np.load('./edge_weights.npy')
    print(edge_weights.shape)

    # print(G.nodes['nft']['word_count:2021_07'])
    # print(G.edges.data())
    # print(G.get_edge_data('blockchain', 'cryptocurrency'))
    # print(G['blockchain']['cryptocurrency']['cooccurrence:2021_07'])

    assert G is not None


def test_get_node_targets():
    output_dir = os.path.abspath('./')
    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    # # \Lib\site-packages\networkx\classes\graph.py
    G = load_graph(graph_file)

    node_indices = pd.read_csv('node_indices.tsv', sep='\t')
    node_names = node_indices['node_name'].tolist()

    with open(os.path.join(output_dir, 'node_attributes.txt'), 'r') as f_nattr:
        node_attrs = f_nattr.readlines()
        targets = []

        for node_name in node_names:
            node_name = node_name.strip()  # 개행문자 제거
            target = []
            for node_attr in node_attrs:
                node_attr = node_attr.strip()
                target.append(G.nodes[node_name][node_attr])
            targets.append(np.asarray(target))
        f_nattr.close()
        np.save('./node_targets.npy', targets)

    node_targets = np.load('./node_targets.npy')
    print(node_targets.shape)





def test_get_edge_indices_and_weights():
    print('\n')

    output_dir = os.path.abspath('./')
    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    G = load_graph(graph_file)

    node_indices = pd.read_csv('node_indices.tsv', sep='\t', names=["node_index", "node_name"])

    with open(os.path.join(output_dir, 'edge_list.txt'), 'r') as f_elist, \
            open(os.path.join(output_dir, 'edge_attributes.txt'), 'r') as f_eattr:
        edge_list = [tuple(map(str, edge.strip().split(' '))) for edge in f_elist]
        edge_attrs = f_eattr.read().splitlines()

        edge_indices_in_time = {}
        edge_weights_in_time = {}

        for edge_attr in edge_attrs[:95]:

            node_u_indices = []
            node_v_indices = []
            edge_weights = []

            for edge in edge_list:
                node_u, node_v = edge
                edge_data = G.get_edge_data(u=node_u, v=node_v, default=0)
                edge_weight = edge_data.get(edge_attr, 0)
                print(node_u, node_v, edge_attr, edge_weight)

                if edge_weight != 0:
                    node_u_index = node_indices[node_indices['node_name'] == node_u]['node_index'].values[0]
                    node_v_index = node_indices[node_indices['node_name'] == node_v]['node_index'].values[0]
                    node_u_indices.append(node_u_index)
                    node_v_indices.append(node_v_index)

                    edge_weights.append(edge_weight)

            edge_indices_in_time[edge_attr] = np.stack((node_u_indices, node_v_indices), axis=0).tolist()
            edge_weights_in_time[edge_attr] = edge_weights

        print(edge_indices_in_time)
        print(edge_weights_in_time)

        with open('edge_indices.json', 'w') as f_i, open('edge_weights.json', 'w') as f_w:
            json.dump(edge_indices_in_time, f_i)
            json.dump(edge_weights_in_time, f_w)


def test_array():
    print('\n')

    a = np.random.rand(3).tolist()
    b = np.random.rand(3).tolist()

    c = np.random.rand(3).tolist()
    d = np.random.rand(3).tolist()

    result = []
    result.append(np.stack((a, b), axis=0).tolist())
    result.append(np.stack((c, d), axis=0).tolist())
    print(result)

def test_dict():
    print('\n')

    dict_ex = {'name': 'song', 'age': 10}

    with open('sample.json', 'w') as f:
        json.dump(dict_ex, f)


