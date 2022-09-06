import argparse
import os
import sys
import pickle
import json
import re
import pandas as pd
import numpy as np
import networkx as nx
import pytest

from network_analysis.coword_detection import parse_preprocessed_data
from network_analysis.graph_reconstruction import reconstruct_graph
from network_analysis.graph_reconstruction import load_graph


# import graph_analysis

def test_make_graph():
    input_f = os.path.abspath('../_datasets/pre_patents_doc.pkl')
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


def test_get_targets():
    output_dir = os.path.abspath('./')
    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    # # \Lib\site-packages\networkx\classes\graph.py
    G = load_graph(graph_file)

    with open(os.path.join(output_dir, 'node_list.txt'), 'r') as f_nname, \
            open(os.path.join(output_dir, 'node_attributes.txt'), 'r') as f_nattr:
        node_names = f_nname.readlines()
        node_attrs = f_nattr.readlines()
        targets = []

        for node_name in node_names:
            node_name = node_name.strip()  # 개행문자 제거
            target = []
            for node_attr in node_attrs:
                node_attr = node_attr.strip()
                target.append(G.nodes[node_name][node_attr])
            targets.append(np.asarray(target))
        f_nname.close()
        f_nattr.close()
        np.save('./node_targets.npy', targets)

    node_targets = np.load('./node_targets.npy')
    print(node_targets.shape)


def test_get_node_edge_list():
    output_dir = os.path.abspath('./')
    graph_file = os.path.join(output_dir, 'combined_graph.pkl')
    G = load_graph(graph_file)

    print(G.nodes)
    with open(os.path.join(output_dir, 'node_list.txt'), 'wb') as f:
        f.write('\n'.join(G.nodes).encode())

    print(G.edges)
    with open(os.path.join(output_dir, 'edge_indices.txt'), 'wb') as f:
        f.write('\n'.join('{} {}'.format(edge[0], edge[1]) for edge in G.edges).encode())
