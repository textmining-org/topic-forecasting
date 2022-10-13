#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import network_utils as net_utils

        
######## Network analysis ########

def _centrality_func_glossary_():
    nx_cent_fns = [
        nx.degree_centrality,
        nx.betweenness_centrality,
        nx.closeness_centrality,
        nx.eigenvector_centrality,
        nx.edge_betweenness_centrality,
    ]
    return {fn.__name__.split('.')[-1]:fn for fn in nx_cent_fns}


# fn: networkx centrality function --> see _centrality_func_glossary_
# returns (dict): {NODE:VALUE}
def _get_centrality_(graph_obj,fn,weight:str)->dict:
    if fn.__name__.split('.')[-1] in ['closeness_centrality']:
        return fn(graph_obj,distance=weight)
    elif fn.__name__.split('.')[-1] in ['degree_centrality']:
        return fn(graph_obj)
    else:
        return fn(graph_obj,weight=weight)


def _connectivity_func_glossary_():
    nx_conn_fns = [
#         nx.all_node_cuts, # w.o. weight
        nx.all_pairs_node_connectivity, # w.o. weight; takes too long
        nx.all_pairs_dijkstra, # wegith 
        nx.all_pairs_bellman_ford_path_length, # weight
        #nx.node_connectivity,
        #nx.resistance_distance,
    ]
    return {fn.__name__.split('.')[-1]:fn for fn in nx_conn_fns}


# Getting connectivity between nodes
# Revised and integrated connectivity analysis function
# fn: networkx connectivity function --> see _connectivity_func_glossary_
# returns all-paired connectivity {NODE1:{NODE2:VALUE}}
def _get_connectivity_(graph_obj,fn,weight:str)->dict:
    # with weight
    fn_name = fn.__name__.split('.')[-1]
    if fn_name == 'all_pairs_node_connectivity':
        return fn(graph_obj)
    if fn_name in ['all_pairs_dijkstra','all_pairs_bellman_ford_path_length']:
        gen = fn(graph_obj,weight=weight)
    # without weight
    else:
        gen = fn(graph_obj)
    conn = {}
    for node1, path_val in gen:
        conn[node1] = {}
        if fn_name in ['all_pairs_dijkstra']:
            for node2, pathlen in path_val[0].items():
                conn[node1][node2] = pathlen
        elif fn_name in ['all_pairs_bellman_ford_path_length']:
            for node2, pathlen in path_val.items():
                conn[node1][node2] = pathlen
    return conn # {NODE:{NODE:VALUE}}
    

def analyze_graph(graph_obj,
                  centrality_function_names:list=['betweenness_centrality'],
                  connectivity_function_names:list=['all_pairs_dijkstra'],
#                   node_weight_keys:list=[],
                  edge_weight_keys:list=[],
                 )->dict:
    cent_gls = _centrality_func_glossary_()
    conn_gls = _connectivity_func_glossary_()
    
    # Centrality analysis
    cent_result = {} # {METHOD_NAME:{ATTRB_KEY:{NODE:VALUE}}}
    for cent_n in centrality_function_names:
        cent_result[cent_n] = {}
        fn = cent_gls[cent_n]
        for wght_key in edge_weight_keys:
            curr_cent = _get_centrality_(graph_obj=graph_obj,fn=fn,weight=wght_key)
            cent_result[cent_n][wght_key] = curr_cent
            
    # Connectivity analysis
    conn_result = {}
    for conn_n in connectivity_function_names:
        conn_result[conn_n] = {}
        fn = conn_gls[conn_n]
        for wght_key in edge_weight_keys:
            curr_conn = _get_connectivity_(graph_obj=graph_obj,fn=fn,weight=wght_key)
            conn_result[conn_n][wght_key] = curr_conn
    
    return {'centrality':cent_result, 'connectivity':conn_result}

