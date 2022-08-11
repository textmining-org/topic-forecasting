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
# import graph_analysis



# Parsing preprocessed data
def parse_preprocessed_data(**kwargs):
    return coword_detection.parse_preprocessed_data(**kwargs)

    
# making network x graph object
def reconstruct_graph(**kwargs):
    return graph_reconstruction.reconstruct_graph(**kwargs)
    
    
# analysis of centrality, connectivity with word count or coword (with inversed value for some cases)
# def analyze_graph(**kwargs):
#     return graph_analysis.analyze_graph(**kwargs)

    

# # keyword_map_file : json file {TOPIC:[KEYWORD]}
# def select_feature(graph_obj_file:str,
#                    keyword_map_file:str,
#                    output_dir:str,
#                   ):
#     output_dir = os.path.abspath(output_dir)
#     os.makedirs(output_dir,exist_ok=True)
    
#     # Graph loading
#     with open(graph_obj_file,'rb') as f:
#         graph_obj = pickle.load(f)
    
#     # keyword map parsing
#     with open(keyword_map_file,'rb') as f:
#         try: # keyword map file is json file
#             keyword_map = json.loads(f.read().decode())
#         except:
#             try:
#                 _r = f.read().decode()
#                 if '\r\n' in _r:
#                     keyword_map = {'Topic':_r.split('\r\n')}
#                 else:
#                     keyword_map = {'Topic':_r.split('\n')}
#                 while '' in keyword_map['Topic']:
#                     keyword_map['Topic'].remove('')
#             except:
#                 raise(AttributeError)
    
#     # feature selection and saving
    
    
    
    
def main():
    print('Initiated')
    parser = argparse.ArgumentParser(description='Process for coword map and graph generation for preprocessed data.')
    
    subparsers = parser.add_subparsers(
        title='Jobs to do',
        description='make_graph / select_feature',
        dest='job',
        help='1. make_grpah : Analysis and making graph for coword map. This job takes preprocessed data as an input file, and generates anlyzed Network x graph object. 2. select_featuer : Selection of features to make data loader. This job takes Network X graph object and make data for GNN training')
    
    ##########################################################
    ######## Parer: Graph reconstruction ana analysis ########
    #########################################################
    
    parser_make_graph = subparsers.add_parser('make_graph', # 1. Analysis
                                              help='Analysis and making graph for coword map')
    
    ######## Arguments for data parsing ########
    parser_make_graph.add_argument('--not_fill_empty_time',
                                   default=True,
                                   action='store_false',
                                   help='Fill empty time line')
    parser_make_graph.add_argument('--input_at_not_document_level',
                                   default=True,
                                   action='store_false',
                                   help='Input file is preprocessed at document level')
    parser_make_graph.add_argument('--no_count_multiplication',
                                   default=True,
                                   action='store_false',
                                   help='multiplicated count for word at a sentece or document')
    parser_make_graph.add_argument('--timeline_normalization',
                                   default=False,
                                   action='store_true',
                                   help='timeline_normalization')
    parser_make_graph.add_argument('--document_normalization',
                                   default=False,
                                   action='store_true',
                                   help='document_normalization')
    parser_make_graph.add_argument('--sentence_normalization',
                                   default=False,
                                   action='store_true',
                                   help='sentence_normalization')
    
    ######## Arguments for graph reconstruction ########
#     parser_make_graph.add_argument('--word_count_annotation_prefix',
#                                    type=str,
#                                    default='word_count',
#                                    help='word_count_annotation_prefix')
    
    
    ######## Arguments for graph analysis ########
    
    
    
    parser_select_feature = subparsers.add_parser('select_feature', # 2. Make data loader
                                                  help='Selection of features to make data loader')
    parser_select_feature.add_argument('--node_val_imputation',
                                       type=float,
                                       required=False,
                                       default=float('nan'),
                                       help='Imputation value for node')
    parser.add_argument('-i','--input',
                       required=True,help='Input file')
    parser.add_argument('-o','--output',
                       required=True,help='Output directory')
    args = parser.parse_args()
    
    if args.job == 'make_graph':
        input_f = os.path.abspath(args.input)
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir,exist_ok=True)
        
        print(f'Parsing preprocessed data... \n{input_f}')
#         parse_preprocessed_data(
#             data_file=args.input,
#             output_dir=args.output,
#             fill_empty_time=args.not_fill_empty_time,
#             input_at_document_level=args.input_at_not_document_level,
#             count_multiplication=args.no_count_multiplication,
#             timeline_normalization=args.timeline_normalization,
#             document_normalization=args.document_normalization,
#             sentence_normalization=args.sentence_normalization,
#         )
        _coword_chunk_file_ = os.path.join(output_dir,'coword_results.pkl')
        print(f'Data parsing has been finished :\n{_coword_chunk_file_}')
        
        print(f'Reconstructing master graph... \n{_coword_chunk_file_}')
        reconstruct_graph(
            coword_file=_coword_chunk_file_,
            word_count_annotation_prefix='word_count', # annotation prefix for node
            whole_word_count_annotation='word_count:whole_time', # annotation of node for whole time
            coword_annotation_prefix='cooccurrence', # annotation prefix for edge
            whole_coword_annotation='cooccurrence:whole_time', # annotation of edge for whole time
            output_dir=output_dir,
        )
        graph_file = os.path.join(output_dir,'combined_graph.pkl')
        node_annotation_file = os.path.join(output_dir,'node_attributes.txt')
        edge_annotation_file = os.path.join(output_dir,'edge_attributes.txt')
        print(f'Master graph reconstruction has been finished :\n{_coword_chunk_file_}')
        
#     elif args.job == 'select_feature':
#         select_feature()
        
#     analyze_coword(data_file=args.input,output_dir=args.output)
    
    
    print('Finished')
    
if __name__=='__main__':
    main()


    