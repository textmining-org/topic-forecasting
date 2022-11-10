#!/usr/bin/env python3
import argparse
import os
import sys
import pickle
import json
import re
import pandas as pd
import numpy as np
import networkx as nx

import coword_detection
import graph_reconstruction
import graph_analysis


# Parsing preprocessed data
def parse_preprocessed_data(**kwargs):
    return coword_detection.parse_preprocessed_data(**kwargs)

    
# making network x graph object
def reconstruct_graph(**kwargs):
    return graph_reconstruction.reconstruct_graph(**kwargs)
    
    
def extract_topic(**kwargs):
    return graph_reconstruction.extract_topic(**kwargs)
    
    
def main():
    print('Initiated')
    parser = argparse.ArgumentParser(description='Process for coword map and graph generation for preprocessed data.')
    
#     #### Common arguments ####
#     parser.add_argument('-i','--input',
#                        required=True,help='Input file or directory')
#     parser.add_argument('-o','--output',
#                        required=True,help='Output directory')
    
    subparsers = parser.add_subparsers(
        title='Jobs to do',
        description='get_coword/make_graph / select_feature',
        dest='job',
        help='''
    1. get_coword : Getting coword map chunk using preprocessed documents
    2. make_grpah : Analysis and making graph for coword map. This job takes preprocessed data as an input file, and generates anlyzed Network x graph object.
    3. extract_topic : Selection of node features as related edge features to make data loader. This job takes Network X graph object and make data for GNN training''')
    
    ##########################################################
    ######## Parer: Graph reconstruction ana analysis ########
    ##########################################################
    
    parser_get_coword = subparsers.add_parser('get_coword', # 1. get_coword
                                              help='Getting coword chunk data using preprocessed document file')
    
    ######## Arguments for get_coword ########
    parser_get_coword.add_argument('-i','--input',
                                   required=True,help='Input file')
    parser_get_coword.add_argument('-o','--output',
                                   required=True,help='Output directory')
    parser_get_coword.add_argument('--not_fill_empty_time',
                                   default=True,
                                   action='store_false',
                                   help='Fill empty time line')
    parser_get_coword.add_argument('--input_at_not_document_level',
                                   default=True,
                                   action='store_false',
                                   help='Input file is preprocessed at document level')
    parser_get_coword.add_argument('--no_count_multiplication',
                                   default=True,
                                   action='store_false',
                                   help='multiplicated count for word at a sentece or document')
    parser_get_coword.add_argument('--timeline_normalization',
                                   default=False,
                                   action='store_true',
                                   help='timeline_normalization')
    parser_get_coword.add_argument('--document_normalization',
                                   default=False,
                                   action='store_true',
                                   help='document_normalization')
    parser_get_coword.add_argument('--sentence_normalization',
                                   default=False,
                                   action='store_true',
                                   help='sentence_normalization')
    parser_get_coword.add_argument('--low_mem',
                                   default=False,
                                   action='store_true',
                                   help='Low memory mode.')
    parser_get_coword.add_argument('--word_count_limit',
                                   default=0,
                                   type=int,
                                   help='Limit on word count for word collection')
    parser_get_coword.add_argument('--without_coword',
                                   default=False,
                                   action='store_true',
                                   help='Without coword mapping, but only with word counting.')
    
    ####### Arguments for graph reconstruction ########
    parser_make_graph = subparsers.add_parser('make_graph', # 2. make_graph
                                              help='Analysis and making graph for coword map')
    parser_make_graph.add_argument('-i','--input',
                                   required=True,help='Input coword result file or directory : result file of \'get_coword\' task')
    parser_make_graph.add_argument('-o','--output',
                                   required=True,help='Output directory')
    parser_make_graph.add_argument('-ct','--centrality',
                                   type=str,
                                   action='append',
                                   default=['betweenness_centrality'],
                                   choices=list(
                                       graph_analysis._centrality_func_glossary_().keys()),
                                   help='Method for centrality')
    
    parser_make_graph.add_argument('-cn','--connectivity',
                                   type=str,
                                   action='append',
                                   default=['all_pairs_dijkstra'],
                                   choices=list(
                                       graph_analysis._connectivity_func_glossary_().keys()),
                                   help='Method for connectivity')
        
#     parser_make_graph.add_argument('--word_count_annotation_prefix',
#                                    type=str,
#                                    default='word_count',
#                                    help='word_count_annotation_prefix')
    
    
    ####### Arguments for extraction of a topic ########
    parser_extract_topic = subparsers.add_parser('extract_topic', # 3. extract_topic
                                              help='Extracting ndoe and edge features for a keyword list')
    parser_extract_topic.add_argument('-i','--input',
                                      required=True,help='Input directory for graphs')
    parser_extract_topic.add_argument('-k','--keyword_file',
                                      required=True,help='File for keywords - tsv or txt formatted == node list')
    parser_extract_topic.add_argument('-t','--time_line_file',
                                      required=True,help='File for timeline - tsv or txt formatted == node list')
    parser_extract_topic.add_argument('-o','--output',
                                      required=True,help='Output directory')
    parser_extract_topic.add_argument('--align_node_order',
                                      default=False,
                                      action='store_true',
                                      help='Align node to central node or defined node')
    parser_extract_topic.add_argument('--central_node',
                                      type=str,
                                      default='blockchain',
                                      help='Pre-defined node to align keywords for node indexing')
    
    args = parser.parse_args()
    
    if args.job == 'get_coword':
        _input = os.path.abspath(args.input)
        _output = os.path.abspath(args.output)
        os.makedirs(_output,exist_ok=True)
        
        print(f'Parsing preprocessed data... \n{_input}')
        parse_preprocessed_data(
            data_file=_input,
            output_dir=_output,
            fill_empty_time=args.not_fill_empty_time,
            input_at_document_level=args.input_at_not_document_level,
            count_multiplication=args.no_count_multiplication,
            timeline_normalization=args.timeline_normalization,
            document_normalization=args.document_normalization,
            sentence_normalization=args.sentence_normalization,
            low_mem_mode=args.low_mem,
            word_count_limit=args.word_count_limit,
            without_coword=args.without_coword,
        )
        _coword_chunk_file_ = os.path.join(_output,'coword_results.pkl')
        print(f'Data parsing has been finished :\n{_coword_chunk_file_}')
    
    elif args.job == 'make_graph':
        _input = os.path.abspath(args.input)
        if os.path.isdir(_input):
            _input = os.path.join(_input,'coword_results.pkl') # inferrence
        _output = os.path.abspath(args.output)
        os.makedirs(_output,exist_ok=True)
        
        print(f'Reconstructing master graph... \n{_input}')
        reconstruct_graph(
            coword_file=_input,
            word_count_annotation_prefix='word_count', # annotation prefix for node
            whole_word_count_annotation='word_count:whole_time', # annotation of node for whole time
            coword_annotation_prefix='cooccurrence', # annotation prefix for edge
            whole_coword_annotation='cooccurrence:whole_time', # annotation of edge for whole time
            output_dir=_output,
            centrality_function_names=args.centrality,
            connectivity_function_names=args.connectivity,
        )
        graph_file = os.path.join(_output,'combined_graph.pkl')
        node_annotation_file = os.path.join(_output,'node_attributes.txt')
        edge_annotation_file = os.path.join(_output,'edge_attributes.txt')
        print(f'Master graph reconstruction has been finished :\n{graph_file}')
        
    elif args.job == 'extract_topic':
        input_package_dir = os.path.abspath(args.input)
        keyword_file = os.path.abspath(args.keyword_file)
        
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir,exist_ok=True)
        
        with open(args.keyword_file,'rb') as f:
            keyword_list=f.read().decode().split()
            while '' in keyword_list:
                keyword_list.remove('')
        with open(args.time_line_file,'rb') as f:
            time_key_list=f.read().decode().split()
            while '' in time_key_list:
                time_key_list.remove('')
        
        result = extract_topic(
            input_package_dir=input_package_dir,
            output_dir=output_dir,
            time_key_list=time_key_list,
            keyword_list=keyword_list,
            cent_methods=['betweenness_centrality','closeness_centrality'],
            conn_methods=['all_pairs_dijkstra'],
            cooc_methods=['inv_cooccurrence'],
            central_node=args.central_node,
            align_node_order=bool(args.align_node_order),
            
        )
        
    print('Finished')
    
if __name__=='__main__':
    main()

