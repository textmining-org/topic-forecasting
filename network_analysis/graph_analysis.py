#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import graph_analysis


# Getting connectivity between nodes
def _get_connectivity_(graph_obj,source=None,target=None):
    connectivity = node_connectivity(G=graph_obj,s=source,t=target)
    return connectivity


# # Getting centrality of a node
# def _get_centrality_(graph_obj,node,attribute_key:str):
#     centrality_map = nx.closeness_centrality(graph_obj,u=node,distance=attribute_key)
#     return centrality_map


# weight: weight on node ?
# distance: distance of edge ?
def _centrality_functions_(graph_obj,
                           weight:str,
                          ):
    # by node
    _degree_centrality_map = nx.degree_centrality(graph_obj)
    _betweenness_centrality_map = nx.betweenness_centrality(graph_obj,weight=weight,normalized=True)
    _closeness_centrality_map = nx.closeness_centrality(graph_obj,distance=weight) #  Absent edge attributes are assigned a distance of 1
    _eigenvector_centrality_map = nx.eigenvector_centrality(graph_obj,max_iter=100,tol=1e-6,weight=weight)
    # by edge
    _edge_betweenness_centrality_map = nx.edge_betweenness_centrality(graph_obj,weight=weight)
    return _degree_centrality_map, _betweenness_centrality_map, _closeness_centrality_map, _eigenvector_centrality_map, _edge_betweenness_centrality_map

# # info_dict: {NODE:ANNOTATION_VALUE}
# def _annotate_node_(graph_obj,info_dict:dict,annotaiton_name:str):
#     for _node, _val in info_dict.items():
#         graph_obj.nodes[_node][annotaiton_name] = _val

# # info_dict: {EDGE:ANNOTATION_VALUE}
# # EDGE: (NODE1,NODE2)
# def _annotate_node_(graph_obj,info_dict:dict,annotaiton_name:str):
#     for _edge, _val in info_dict.items():
#         graph_obj.edges[_edge][annotaiton_name] = _val
        
        
def _inverse_(_v):
    return 1.0/_v

def _make_ratio_(_v,whole_sum):
    return _v/whole_sum


def _modify_annotation_(graph_sub_obj,
                        modification_function,
                        target_annotation_key:str,prefix:str='inversed',drop_zero=True):
    _annot_n = ':'.join([prefix,target_annotation])
    for _nd_or_edg, _annot in graph_sub_obj.items():
        if target_annotation in _annot:
            if drop_zero and _annot[target_annotation]:
                continue
            else:
                graph_sub_obj[_nd_or_edg][_annot_n] = modification_function(_annot[target_annotation])
    return graph_sub_obj
        
# def _inverse_annotation_(graph_sub_obj,target_annotation_key:str,prefix:str='inversed',drop_zero=True):
#     _annot_n = ':'.join([prefix,target_annotation])
#     for _nd_or_edg, _annot in graph_sub_obj.items():
#         if target_annotation in _annot:
#             if drop_zero and _annot[target_annotation]:
#                 continue
#             else:
#                 graph_sub_obj[_nd_or_edg][_annot_n] = 1.0/_annot[target_annotation]
#     return graph_sub_obj


# edge_annotation_name: [ANNOTATION_NAME1,...] annotation names of edges to extract
# nodes : [NODE] Nodes for subgraph. If nodes are not implemented, isolated nodes will be remove.
def _extract_annotation_(graph_obj,
                         node_annotation_name:list=[],
                         edge_annotation_name:list=['cooccurrence:whole_time'],
                         target_nodes:list=None,
                        ):
    sub_g = nx.Graph()
    target_edges = []
    for _e, annots in graph_obj.edges.items():
        if set(edge_annotation_name) & set(annots.keys()):
            target_edges.append((_e, {_k:_v for _k,_v in annots.items() if _k in edge_annotation_name}))
            
    if not target_nodes:
        target_nodes = []
        for edge_info,annot_dict in target_edges:
            target_nodes.extend([edge_info[0],edge_info[1]])
        target_nodes = list(set(target_nodes))
    else:
        # remove irrelated edges with node - if node list is given
        _target_edges_ = []
        for _e, annot in target_edges:
            if _e[0] in target_nodes and _e[1] in target_nodes:
                _target_edges_.append((_e,annot))
        target_edges = _target_edges_
    
    for _n in target_nodes:
        sub_g.add_node(_n)
        for _k, _v in G.nodes[_n].items():
            if _k in node_annotation_name:
                sub_g.nodes[_n][_k] = _v
                
    for _e, annot in target_edges:
        sub_g.add_edges_from([_e])
        for _k, _v in annot.items():
            sub_g.edges[_e][_k] = _v
    return sub_g
    


def analyze_graph(graph_obj,
                  node_annotations_to_inv:list,
                  edge_annotations_to_inv:list,
                  drop_inversed_zero:bool=True):
    # inversed node annotation values
    for _annot_name in node_annotations_to_inv:
        _modify_annotation_(
            graph_obj.nodes,
            modification_function=_inverse_,
            target_annotation=_annot_name,
            prefix='inversed',drop_zero=drop_inversed_zero)
    # inversed edge annotation values
    for _annot_name in edge_annotations_to_inv:
        _modify_annotation_(
            graph_obj.edges,
            modification_function=_inverse_,
            target_annotation=_annot_name,
            prefix='inversed',drop_zero=drop_inversed_zero)
    
    
    
    # centrality for node
    
    
    # detection hub node
    
    
    # connectiviy for edges
    
    
    
    
    return graph_obj


        
        
# convert graph to df
# df.loc[i,i] is graph_obj's node's annotation.
def convert_graph_to_df(graph_obj,
                        node_list:list,
                        node_val_key:str='',
                        edge_val_key:str='',
                        node_val_imputation=float('nan'),
                        edge_val_imputation=float('nan'),
                       ):
    df = pd.DataFrame({},index=node_list,columns=node_list)
    for _n in node_list:
        if node_val_key in graph_obj.nodes[_n]:
            df.loc[_n,_n] = graph_obj.nodes[_n][node_val_key]
        else:
            df.loc[_n,_n] = node_val_imputation
    for s, t in itertools.permutations(node_list,2):
        if edge_val_key in graph_obj.edges[s,t]:
            df.loc[s,t] = graph_obj.edges[s,t][edge_val_key]
        else:
            df.loc[s,t] = edge_val_imputation
        
    return df
