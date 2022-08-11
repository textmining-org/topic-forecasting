import argparse
import os
import sys
import pickle
import json
import re
import pandas as pd
import numpy as np
import networkx as nx

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
    reconstruct_graph(
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
    G = load_graph(graph_file)

    # \Lib\site-packages\networkx\classes\graph.py
    print(G.nodes)
    print(G.edges)

    assert G is not None
