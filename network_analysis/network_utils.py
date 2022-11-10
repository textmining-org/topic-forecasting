#!/usr/bin/env python3
import os
import re
import pickle
import json
import itertools
import pandas as pd
import numpy as np
import networkx as nx


        
def _inverse_(_v:float):
    if not _v:
        return float('inf')
    else:
        return 1.0/_v

    
def _make_ratio_(_v,whole_sum):
    return _v/whole_sum


#########################
######## Data IO ########
#########################

def parse_list(file):
    with open(file,'rb') as f:
        _ = f.read().decode().split('\n')
        while '' in _:
            _.remove('')
    return _


def write_list(data:list,file):
    with open(file,'wb') as f:
        f.write(('\n'.join(data)).encode())

        
def append_list(data:list,file):
    with open(file,'a') as f:
        f.write('\n')
        f.write('\n'.join(data))
        

def parse_json(file):
    with open(file,'rb') as f:
        _ = json.loads(f.read().decode())
    return _
    
    
def write_json(data,file):
    with open(file,'wb') as f:
        f.write(json.dumps(data).encode())

        
def parse_lowmem_coword_dict(file):
    with open(file,'rb') as f:
        _ = json.loads(f.read().decode())
    return {tuple(_k.split(':')):_v for _k, _v in _.items()}
        
def write_lowmem_coword_dict(data,file): # {(WORD1,WORD2):FLOAT}
    with open(file,'wb') as f:
        f.write(json.dumps({':'.join(_k):_v for _k, _v in data.items()}).encode())
    

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
    
    
def _read_node_index_(file):
    return pd.Series(pd.read_csv(file,header=None,index_col=0,sep='\t').iloc[:,0])
    
    
# node_ser: pd.Series index: 0 ~ n. values: word
def _write_node_index_(node_ser,output='./node_indices.tsv'):
    node_ser.to_csv(output,sep='\t',header=None)
    
    
def _read_edge_index_(file):
    with open(file,'rb') as f:
        _raw_edge_list_ = f.read().decode().split('\n')
    edge_list = [tuple(i.split('\t')) for i in _raw_edge_list_ if '\t' in i]
    return edge_list
    
    
# node_list: list of nodes - if edge_list is not given, node_list-inferred edge_list would be written.
# edge_list: list of edge connection [(NODE1,NODE2),(NODE1,NODE3),...]
def _write_edge_index_(node_list=[],edge_list=[],output='./edge_list.tsv'):
    if not edge_list and node_list:
        edge_list = list(itertools.combinations(node_list,2))
    with open(output,'w') as f:
        for n1, n2 in edge_list:
            f.write(n1+'\t'+n2+'\n')
    
    
# features: (gcn_io_format==False) {ATTRIBUTE:{NODE:VALUE}}
# features: (gcn_io_format==True) 2-dim np.array - dim1 : time(or attribute)-specific, dim2 : node-specific
def _read_node_features_(file,gcn_io_format=True):
    if gcn_io_format:
        return np.load(file)
    else:
        with open(file, 'rb') as f:
            feature = json.loads(f.read().decode())
        return feature
    
    
# node_features(gcn_io_format==False): {ATTRIBUTE:{NODE:VALUE}}
# node_features(gcn_io_format==True): 2-dim np.array - dim1 : time(or attribute)-specific, dim2 : node-specific
def _write_node_features_(node_features,file,gcn_io_format=True):
    if gcn_io_format:
        np.save(file=file,arr=np.array(node_features))
    else:
        with open(file,'wb') as f:
            f.write(json.dumps(file))
    
    
def _read_edge_features_(input_prefix,gcn_io_format=True):
    if gcn_io_format:
        if os.path.isdir(input_prefix):
            edge_idx_f = os.path.join(input_prefix,'edge_indices.json')
            edge_wgt_f = os.path.join(input_prefix,'edge_weigths.json')
        else:
            edge_idx_f = input_prefix+'.edge_indices.json'
            edge_wgt_f = input_prefix+'.edge_weigths.json'
        with open(edge_idx_f, 'rb') as f:
            idx_d = json.loads(f.read().decode())
        with open(edge_wgt_f, 'rb') as f:
            wgt_d = json.loads(f.read().decode())
            
        feats = {}
        for _attrb, [src_l,tar_l] in idx_d.items():
            feats[_attrb] = {}
            for _idx, src_n in enumerate(src_l):
                if src_n not in feats[_attrb]:
                    feats[_attrb][src_n] = {}
                feats[_attrb][src_n][tar_l[_idx]] = wgt_d[_attrb][_idx]
        return feats
    else:
        with open(input_prefix,'rb') as f:
            feats = json.loads(f.read().decode())
        return feats
    
# edge_features: (gcn_io_format==False) {ATTRIBUTE:{SOURCE_NODE:{TARGET_NODE:VALUE}}}
# output shape(gcn_io_format==True, ~/edge_indices.json) {ATTRIBUTE:[[SOURCE_NODE_IDX],[TARGET_NODE_IDX]]}
# output shape(gcn_io_format==True, ~/edge_weights.json) {ATTRIBUTE:[VALUE]}
def _write_edge_features_(edge_features:dict,output_prefix,gcn_io_format=True):
    if gcn_io_format:
        if os.path.isdir(output_prefix):
            edge_idx_f = os.path.join(output_prefix,'edge_indices.json')
            edge_wgt_f = os.path.join(output_prefix,'edge_weigths.json')
        else:
            edge_idx_f = output_prefix+'.edge_indices.json'
            edge_wgt_f = output_prefix+'.edge_weigths.json'
        
        edge_idx_d = {}
        edge_wgt_d = {}
        for _attrb, edges in edge_features.items():
            edge_idx_d[_attrb] = [[],[]]
            edge_wgt_d[_attrb] = []
            for n1, target_val_d in edges.items():
                edge_idx_d[_attrb][0].extend([n1]*len(list(target_val_d.keys())))
                edge_idx_d[_attrb][1].extend(list(target_val_d.keys()))
                edge_wgt_d[_attrb].extend(list(target_val_d.values()))
            assert len(edge_idx_d[_attrb][0]) == len(edge_idx_d[_attrb][1])
            assert len(edge_idx_d[_attrb][0]) == len(edge_wgt_d[_attrb])
        assert set(edge_idx_d.keys()) == set(edge_wgt_d.keys())
        
        with open(edge_idx_f,'wb') as f:
            f.write(json.dumps(edge_idx_d).encode())
        with open(edge_wgt_f,'wb') as f:
            f.write(json.dumps(edge_wgt_d).encode())
    else:
        with open(output,'wb') as f:
            f.write(json.dumps(edge_features).encode())
    
    
#########################
###### Conversion #######
#########################

def _extr_kw_at_node_(keyword_list:list, # [NODE1, NODE2, ...]
                      node_features:dict, # {NODE:VALUE}
                      imputation=None,
                     )->dict:
    if imputation==None:
        subportion = {i:node_features[i] for i in keyword_list if i in node_features}
    else:
        _impt = lambda nd, ftr_d, impt: ftr_d[nd] if nd in ftr_d else impt
        subportion = {i:_impt(i,node_features,imputation) for i in keyword_list}
    return subportion


def _extr_kw_at_edge_(keyword_list:list, 
                      edge_features:dict, # {NODE:{NODE:VALUE}}
                      imputation=None,
                     )->dict:
    if imputation==None:
        subportion = {i:{
            j:edge_features[i][j] for j in keyword_list if j in edge_features[i] and i!=j
        } for i in keyword_list if i in edge_features}
    else:
        subportion = {}
        _impt = lambda nd, ftr_d, impt: ftr_d[nd] if nd in ftr_d else impt
        for _n1 in keyword_list:
            if _n1 in edge_features:
                subportion[_n1] = {i:_impt(i,edge_features[_n1],imputation) for i in keyword_list if i!=_n1 and _n1 in edge_features}
            else:
                subportion[_n1] = {i:imputation for i in keyword_list if i!=_n1}
    return subportion


def _cnvrt_node_ftr_(word_idx_map, # {WORD:MAP_IDX}
                     node_features, # {NODE:VALUE}
                    )->dict:
    converted = {word_idx_map[_wrd]:_val for _wrd, _val in node_features.items()}
    return converted
    
    
def _cnvrt_edge_ftr_(word_idx_map, # {WORD:MAP_IDX}
                     edge_features, # {NODE:{NODE:VALUE}}
                    )->dict:
    converted = {word_idx_map[_wrd]:{
        word_idx_map[_sub_wrd]:_val for _sub_wrd, _val in _sub_val.items()
    } for _wrd, _sub_val in edge_features.items()}
    return converted


# node_feature_dict: {ATTRB:{NODE:VAL}}
# edge_feature_dict: {ATTRB:{NODE1:{NODE2:VAL}}}
def convert_feature_dicts(keyword_list:list,
                          node_feature_dict:dict,
                          edge_feature_dict:dict,
                         )->(dict,dict):
    word_idx_map = {node_n:idx for idx,node_n in enumerate(keyword_list)}
    converted_node_features = {attrb:_cnvrt_node_ftr_(
        word_idx_map=word_idx_map,
        node_features=nd_fts,
    ) for attrb, nd_fts in node_feature_dict.items()}
    converted_edge_features = {attrb:_cnvrt_edge_ftr_(
        word_idx_map=word_idx_map,
        edge_features=eg_fts,
    ) for attrb, eg_fts in edge_feature_dict.items()}
    return converted_node_features, converted_edge_features


# NOTE: attribute key for nodes and edges should be solely existed
# returns list of dicts : node_attrbs={ATTRB_KEY:{NODE:VALUE}}
def extract_keyword_features(graph_obj,
                             keyword_list:list, # [NODE1, NODE2, ...]
                             node_attribute_key:str='word_count',
                             edge_attribute_key:str='inv_cooccurrence',
                             centrality_results:dict={}, # {NODE:VALUE}
                             connectivity_results:dict={}, # {NODE1:{NODE2:VALUE}}
                            ):
    _node_ref = {n:graph_obj.nodes[n] for n in keyword_list if n in graph_obj.nodes}
    node_attrbs = {n:n_attrb[node_attribute_key] for n, n_attrb in _node_ref.items() if node_attribute_key in n_attrb}
    _edge_ref = {e:graph_obj.edges[e] for e in itertools.combinations(keyword_list,2) if e in graph_obj.edges}
    edge_attrbs = {e:e_attrb[edge_attribute_key] for e, e_attrb in _edge_ref.items() if edge_attribute_key in e_attrb}
    cent_portions = _extr_kw_at_node_(
        keyword_list=keyword_list,
        centrality_dict=centrality_results,
        imputation=0.0
    )
    conn_portions = _extr_kw_at_edge_(
        keyword_list=keyword_list,
        connectivity_dict=connectivity_results,
        imputation=None,
    )
    
    return node_attrbs, edge_attrbs, cent_portions, conn_portions
    
    
def _impute_features_d1_(data_dict,ref_list,imputaion):
    _imput_target = set(ref_list)-set(data_dict.keys())
    _d_d = data_dict.copy()
    for _k in _imput_target:
        _d_d[_k] = imputaion
    return _d_d


def _impute_features_d2_(data_dict,ref_list,imputaion):
    _imput_target = set(ref_list)-set(data_dict.keys())
    _d_d = data_dict.copy()
    for _k in _imput_target:
        _d_d[_k] = {}
    curr_dd_ks = list(_d_d.keys())
    for _k in curr_dd_ks:
        _d_d[_k] = _impute_features_d1_(
            data_dict=_d_d[_k],ref_list=ref_list,imputation=imputation)
    return _d_d


# rule1: edge_features should contain node-indexed edges (e.g. (i,j) where i and j are int)
# rule2: i<j
# NOTE! rule3: duplicated edge values are randomly chosen - apply this function at only case for diagonally symmetric matrix
# edge_featuers = {NODE1:{NODE2:VAL}}
def _make_nonredundancy_for_edges_(edge_features):
    n_edge_ftrs = {}
    for _n1, _n2_val_d in edge_features.items():
        for _n2, _val in _n2_val_d.items():
            if _n1<_n2:
                if _n1 not in n_edge_ftrs:
                    n_edge_ftrs[_n1] = {}
                n_edge_ftrs[_n1][_n2] = _val
            else:
                if _n2 not in n_edge_ftrs:
                    n_edge_ftrs[_n2] = {}
                n_edge_ftrs[_n2][_n1] = _val
    return n_edge_ftrs

#########################
#### Data Conversion ####
#########################

    
    
# # convert graph to df
# # df.loc[i,i] is graph_obj's node's annotation.
# def convert_graph_to_df(graph_obj,
#                         node_list:list,
#                         node_val_key:str='',
#                         edge_val_key:str='',
#                         node_val_imputation=float('nan'),
#                         edge_val_imputation=float('nan'),
#                        ):
#     df = pd.DataFrame({},index=node_list,columns=node_list)
#     for _n in node_list:
#         if node_val_key in graph_obj.nodes[_n]:
#             df.loc[_n,_n] = graph_obj.nodes[_n][node_val_key]
#         else:
#             df.loc[_n,_n] = node_val_imputation
#     for s, t in itertools.permutations(node_list,2):
#         if edge_val_key in graph_obj.edges[s,t]:
#             df.loc[s,t] = graph_obj.edges[s,t][edge_val_key]
#         else:
#             df.loc[s,t] = edge_val_imputation
        
#     return df


# def _get_node_props_to_dict_(graph_obj,
#                              node_attrb_key:str,
#                              imputation:float=None,
#                              node_list:list=None, # In case to specify nodes
#                             ):
#     if not node_list:
#         node_list = [i for i in graph_obj.nodes if node_attrb_key in graph_obj.nodes[i]]
#     if imputation:
#         _fn = lambda i, attrb_k, imputation: graph_obj.nodes[i][attrb_k] if attrb_k in graph_obj.nodes[i] else imputation
#         node_d = {i:_fn(i,node_attrb_key,imputation) for i in node_list}
#     else:
#         node_d = {i:graph_obj.nodes[i][node_attrb_key] for i in node_list}
    
#     return node_d


# def convert_graph_to_dict(graph_obj,
#                           node_attrb_key:str,
#                           edge_attrb_key:str,
#                           imputation:float=None,
#                           node_list:list=None, # In case to specify nodes
#                           edge_list:list=None, # In case to specify edges
#                          ):
    
    
    
#     if not edge_list:
#         edge_list = [i for i in graph_obj.edges if edge_attrb_key in graph_obj.edges[i]]
#     edge_d = {i:graph_obj.edges[i][edge_attrb_key] for i in edge_list}
    
#     edge_infrd_node_l = []
#     for _e in edge_d:
    
#     assert set(node_list) == set(edge_infrd_node_l)
    
#     return node_d, edge_d
    



