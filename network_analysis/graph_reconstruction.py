#!/usr/bin/env python3
import os
import re
import json
import itertools
import pickle
import pandas as pd
import networkx as nx

#################################
######## Network X graph ########
#################################

#### Graph reconstruction using coword dictionaries ####

# Make coword graph template : addition of nodes
def make_coword_graph(coword_dict:dict={},word_list:list=[]):
    if not word_list:
        word_list = []
        for _w1, _w2 in coword_dict:
            word_list.extend([_w1,_w2])
        word_list = sorted(list(set(word_list)))
    
    G = nx.Graph()
    G.add_nodes_from(word_list)
    
    return G


# attribution_dict: {NODE_ID:VALUE} or {NODE_ID:{KEY:VALUE}}
# IF attribution_dict = {NODE_ID:VALUE}, "key" should be imposed as an argument
# if add_weight_val: annotated key's value would be added
def _annotate_node_(graph_obj,attribution_dict:dict,key:str=False, val_addition:bool=False):
    for _n, _v in attribution_dict.items():
        if key:
            if val_addition:
                if key in graph_obj.nodes[_n]:
                    graph_obj.nodes[_n][key] += _v
                else:
                    graph_obj.nodes[_n][key] = _v
            else:
                graph_obj.nodes[_n][key] = _v
        else:
            if val_addition:
                for _k, _sub_v in _v.items():
                    if _k in graph_obj.nodes[_n]:
                        graph_obj.nodes[_n][_k] += _sub_v
                    else:
                        graph_obj.nodes[_n][_k] = _sub_v
            else:
                graph_obj.nodes[_n].update(_v)
    return graph_obj
    
    
# Annotate edges with weight or capacity
def _annotate_edge_(graph_obj, coword_dict:dict,
                    attribute_name='weight', val_addition:bool=False):
    if val_addition:
        for (_w1, _w2), weight_val in coword_dict.items():
            graph_obj.edges[_w1,_w2][attribute_name] += weight_val
    else:
        for (_w1, _w2), weight_val in coword_dict.items():
            graph_obj.edges[_w1,_w2][attribute_name] = weight_val
    return graph_obj
    
# node_list: [WORD] - whole words
# edge_list: [(WORD1,WORD2)] - whole edges
# node_dict_dict : {ATTRIBUTE_NAME:ANNOTATION_DICT} - ANNOTATION_DICT = {WORD:VALUE}
# edge_dict_dict : {ATTRIBUTE_NAME : COWORD_DICT} - COWORD_DICT = {(WORD1,WORD2):VALUE}
def _reconstruct_graph_(node_list:list,
                        node_dict_dict:dict,
                        edge_list:list,
                        edge_dict_dict:dict):
    G = make_coword_graph(word_list=node_list)
    for _k, annot_dict in node_dict_dict.items():
        _annotate_node_(graph_obj=G,
                        attribution_dict=annot_dict,
                        key=_k,
                        val_addition=False)
    G.add_edges_from(edge_list)
    for _k, coword_dict in edge_dict_dict.items():
        _annotate_edge_(graph_obj=G,
                        coword_dict=coword_dict,
                        attribute_name=_k,
                        val_addition=False)
    
    return G


# coword_file: file of coword map - if imposed, word_list, word counts and coword maps are ignored.
# word_list : [WORD] node_list. - whole words
# whole_word_count:dict, # {WORD:COUNT}
# by_month_word_count:dict, # {TIME:{WORD:COUNT}}
# whole_coword_map:dict, # {(WORD1,WORD2):COUNT}
# by_month_coword_map:dict, # {TIME:{(WORD1,WORD2):COUNT}}
# word_count_annotation_prefix:str='word_count:', # annotation prefix for node
#whole_word_count_annotation:str='word_count:whole_time', # annotation of node for whole time
#coword_annotation_prefix:str='cooccurrence:', # annotation prefix for edge
#whole_coword_annotation:str='cooccurrence:whole_time', # annotation of edge for whole time
def reconstruct_graph(coword_file:str=None,
                      word_list:list=None, # [WORD]
                      whole_word_count:dict=None, # {WORD:COUNT}
                      by_month_word_count:dict=None, # {TIME:{WORD:COUNT}}
                      whole_coword_map:dict=None, # {(WORD1,WORD2):COUNT}
                      by_month_coword_map:dict=None, # {TIME:{(WORD1,WORD2):COUNT}}
                      word_count_annotation_prefix:str='word_count', # annotation prefix for node
                      whole_word_count_annotation:str='word_count:whole_time', # annotation of node for whole time
                      coword_annotation_prefix:str='cooccurrence', # annotation prefix for edge
                      whole_coword_annotation:str='cooccurrence:whole_time', # annotation of edge for whole time
                      output_dir:str='./output',
                     )->('nx.Graph', list, list):
    # In case that coword file is given
    if coword_file:
        word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map = parse_coword_chunk(coword_file)
        
    o_d = os.path.abspath(output_dir)
    
    if word_count_annotation_prefix:
        _node_dict = {':'.join(
            [word_count_annotation_prefix,_k]):_v for _k, _v in by_month_word_count.items()}
    else:
        _node_dict = by_month_word_count.copy()
    _node_dict.update({whole_word_count_annotation:whole_word_count})
    
    if coword_annotation_prefix:
        _edge_dict = {':'.join(
            [coword_annotation_prefix,_k]):_v for _k, _v in by_month_coword_map.items()}
    else:
        _edge_dict = by_month_coword_map.copy()
    _edge_dict.update({whole_coword_annotation:whole_coword_map})
    edge_list = list(whole_coword_map.keys())
    G = _reconstruct_graph_(node_list=word_list,
                            node_dict_dict=_node_dict,
                            edge_list=edge_list,
                            edge_dict_dict=_edge_dict,
                           )
    # Annotated attribute titles
    node_annotations = list(_node_dict.keys())
    edge_annotations = list(_edge_dict.keys())
    
    # Save
    save_graph(graph_obj=G,output_file=os.path.join(o_d,'combined_graph.pkl'))
    with open(os.path.join(o_d,'node_attributes.txt'),'wb') as f:
        f.write('\n'.join(node_annotations).encode())
    with open(os.path.join(o_d,'edge_attributes.txt'),'wb') as f:
        f.write('\n'.join(edge_annotations).encode())
    
    return G, node_annotations, edge_annotations
    

#########################
######## Data IO ########
#########################

def parse_coword_chunk(data_pkl_file):
    with open(data_pkl_file,'rb') as f:
        coword_chunk = pickle.load(f)
    return coword_chunk # word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map


def save_graph(graph_obj,output_file):
    with open(output_file,'wb') as f:
        pickle.dump(graph_obj,f)

def load_graph(graph_file):
    with open(graph_file,'rb') as f:
        graph_obj = pickle.load(f)
    return graph_obj
    
