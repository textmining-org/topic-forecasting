#!/usr/bin/env python3
import os
import re
import json
import itertools
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import graph_analysis
import network_utils as net_utils


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

#################################################
######## subgraphs and analysis by month ########
#################################################

# Nodes without any value at the the time_key are excluded.
def _recon_time_graph_(by_month_word_count:dict, # {TIME_KEY:{NODE_KEY:VALUE}}
                       by_month_coword_map:dict, # {TIME_KEY:{EDGE_KEY:VALUE}}
                       time_key:str='2022_01', # TIME_KEY - should be annoted key at by_month_word_count and by_month_coword_map
                       word_attrb_name:str='word_count', # Intended annotation key for nodes
                       cooccr_attrb_name:str='cooccurrence', # Intended annotation key for edges
                       inverse_cooccurrence_value:bool=False, # To inverse edge value
                      ):
    
    edge_list = list(by_month_coword_map[time_key].keys())
    if inverse_cooccurrence_value:
        edge_dict = {cooccr_attrb_name:{
            _edge:1./_val for _edge, _val in by_month_coword_map[time_key].items() if _val}} # Dropping edge_val == 0.0
    else:
        edge_dict = {cooccr_attrb_name:{
            _edge:_val for _edge, _val in by_month_coword_map[time_key].items() if _val}} # Dropping edge_val == 0.0
    node_list = [wrd for wrd in by_month_word_count[time_key] if by_month_word_count[time_key][wrd]] # Dropping node value==0.0
    node_dict = {word_attrb_name:{n:by_month_word_count[time_key][n] for n in node_list}}
    
    G = _reconstruct_graph_(
        node_list=node_list,
        node_dict_dict=node_dict,
        edge_list=edge_list,
        edge_dict_dict=edge_dict,
    )
    return G


def reconstruct_time_graphs(by_month_word_count:dict=None,
                            by_month_coword_map:dict=None,
                            coword_file:str=None,
                            output_dir:str=None,
                           ):
    if (not by_month_word_count or not by_month_coword_map) and coword_file:
        word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map = net_utils.parse_coword_chunk(coword_file)
    sub_Gs = {}
    if output_dir:
        output_dir = os.path.abspath(outupt_dir)
        os.makedirs(output_dir,exit_ok=True)
    for inv_bool_flt, cooc_inv_name in enumerate(['cooccurrence','inv_cooccurrence']):
        sub_Gs[cooc_inv_name] = {}
        for time_key in by_month_word_count.keys():
            G = _recon_time_graph_(
                by_month_word_count=by_month_word_count,
                by_month_coword_map=by_month_coword_map,
                time_key=time_key,
                word_attrb_name='word_count',
                cooccr_attrb_name=cooc_inv_name,
                inverse_cooccurrence_value=bool(inv_bool_flt),
            )
            sub_Gs[cooc_inv_name][time_key] = G
            if output_dir:
                net_utils.save_graph(
                    graph_obj=G,
                    output_file=os.path.join(output_dir,f"graph.{cooc_inv_name}.{time_key}.pkl"),
                )
    return sub_Gs



######## Master Function ########

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
                      centrality_function_names:list=['betweenness_centrality','closeness_centrality'],
                      connectivity_function_names:list=['all_pairs_dijkstra'],
                     )->('nx.Graph', list, list):
    # In case that coword file is given
    if coword_file:
        print(f"Parsing file...\t{coword_file}")
        print(net_utils.parse_coword_chunk(coword_file))
        word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map = net_utils.parse_coword_chunk(coword_file)
        
    o_d = os.path.abspath(output_dir)
    os.makedirs(o_d,exist_ok=True)
    
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
    print('Reconstructing master graph network...')
    G = _reconstruct_graph_(node_list=word_list,
                            node_dict_dict=_node_dict,
                            edge_list=edge_list,
                            edge_dict_dict=_edge_dict,
                           )
    # Annotated attribute titles
    node_annotations = list(_node_dict.keys())
    edge_annotations = list(_edge_dict.keys())
    
    # Save
    net_utils.save_graph(graph_obj=G,output_file=os.path.join(o_d,'combined_graph.pkl'))
    with open(os.path.join(o_d,'node_attributes.txt'),'wb') as f:
        f.write('\n'.join(node_annotations).encode())
    with open(os.path.join(o_d,'edge_attributes.txt'),'wb') as f:
        f.write('\n'.join(edge_annotations).encode())
        
    ## whole_time G analysis
#     # Graph analysis - whole time cooccurrence and inv_cooccurrence
#     print('Analyzing whole-time graph network...')
#         # Cooccurrence
#     cooc_G = _recon_time_graph_(
#         by_month_word_count={'whole_time':whole_word_count},
#         by_month_coword_map={'whole_time':whole_coword_map},
#         time_key='whole_time',
#         word_attrb_name='word_count',
#         cooccr_attrb_name='cooccurrence',
#         inverse_cooccurrence_value=False,
#     )
    
#     cooc_analyzed = graph_analysis.analyze_graph(
#         graph_obj=cooc_G,
#         centrality_function_names=centrality_function_names,
#         connectivity_function_names=connectivity_function_names,
#         edge_weight_keys=['cooccurrence'],
#     )
#     net_utils.save_graph(graph_obj=cooc_G,output_file=os.path.join(o_d,'whole_time.cooccurrence.graph.pkl'))
#     with open(os.path.join(o_d,f'whole_time.cooccurrence.centrality.json'),'wb') as f:
#         f.write(json.dumps(cooc_analyzed['centrality']).encode())
#     with open(os.path.join(o_d,f'whole_time.cooccurrence.connectivity.json'),'wb') as f:
#         f.write(json.dumps(cooc_analyzed['connectivity']).encode())
        
#         # Inversed cooccurrence
#     inv_cooc_G = _recon_time_graph_(
#         by_month_word_count={'whole_time':whole_word_count},
#         by_month_coword_map={'whole_time':whole_coword_map},
#         time_key='whole_time',
#         word_attrb_name='word_count',
#         cooccr_attrb_name='inv_cooccurrence',
#         inverse_cooccurrence_value=True,
#     )
#     inv_cooc_analyzed = graph_analysis.analyze_graph(
#         graph_obj=inv_cooc_G,
#         centrality_function_names=centrality_function_names,
#         connectivity_function_names=connectivity_function_names,
#         edge_weight_keys=['inv_cooccurrence'],
#     )
#     net_utils.save_graph(graph_obj=inv_cooc_G,output_file=os.path.join(o_d,'whole_time.inv_cooccurrence.graph.pkl'))
#     with open(os.path.join(o_d,f'whole_time.inv_cooccurrence.centrality.json'),'wb') as f:
#         f.write(json.dumps(inv_cooc_analyzed['centrality']).encode())
#     with open(os.path.join(o_d,f'whole_time.inv_cooccurrence.connectivity.json'),'wb') as f:
#         f.write(json.dumps(inv_cooc_analyzed['connectivity']).encode())
        
    # Subgraphs - timeline-specific
    sub_g_o_d = os.path.join(o_d,'time_speicific_graphs')
    print(f"Output directory for sub graphs and analysis results: {sub_g_o_d}")
    os.makedirs(sub_g_o_d,exist_ok=True)
    sub_Gs = reconstruct_time_graphs(
        by_month_word_count=by_month_word_count,
        by_month_coword_map=by_month_coword_map,
        output_dir=False,
    )
    
    for cooc_key, t_sub_G_d in sub_Gs.items():
        for time_key, sub_G in t_sub_G_d.items():
            print(f'Current analysis: {time_key}.{cooc_key}')
            net_utils.save_graph(
                graph_obj=sub_G,
                output_file=os.path.join(sub_g_o_d,f'{time_key}.{cooc_key}.graph.pkl'))
            assert os.path.isfile(os.path.join(sub_g_o_d,f'{time_key}.{cooc_key}.graph.pkl'))
            # _t_analyzed = {'centrality':{CENT_FUNC_NAME:{EDGE_WEIGHT_KEY:{NODE:VAL}}},
            #     'connectivity':{CONN_FUNC_NAME:{EDGE_WEIGHT_KEY:{NODE:{NODE:VAL}}}}}
            # NOTE: EDGE_WEIGHT_KEY is dependent on cooc vs inv_coor
            _t_analyzed = graph_analysis.analyze_graph(
                graph_obj=sub_G,
                centrality_function_names=centrality_function_names,
                connectivity_function_names=connectivity_function_names,
                edge_weight_keys=[cooc_key],
            )
            with open(os.path.join(sub_g_o_d,f'{time_key}.{cooc_key}.centrality.json'),'wb') as f:
                f.write(json.dumps(_t_analyzed['centrality']).encode())
            with open(os.path.join(sub_g_o_d,f'{time_key}.{cooc_key}.connectivity.json'),'wb') as f:
                f.write(json.dumps(_t_analyzed['connectivity']).encode())
    
    return G, node_annotations, edge_annotations
    
    
######## Extraction of keyword-based subgraph properties ########
    
    
# realign keyword list based on central_node-cooccurrence
def _realign_keyword_(whole_coword_map:dict,central_node:str,keyword_list:list):
    _sel_foo = lambda x,y,false_val: x if x!=false_val else y
    _sub_cowrd_d_ = {_sel_foo(n1,n2,central_node):val for (n1,n2), val in whole_coword_map.items() if central_node in [n1,n2]}
    _sub_coword_ser = pd.Series(_sub_cowrd_d_,index=keyword_list)
    _sub_coword_ser.fillna(0.0,inplace=True)
    aligned_keyword_list = list(_sub_coword_ser.sort_values(ascending=False).index)
    return aligned_keyword_list
    
    
def extract_topic(input_package_dir:str,
                  output_dir:str,
                  time_key_list:list, # [TIME1,TIME2] <- must be ordered
                  whole_time_graph_file:str=None, # can be inferred by input_package_dir
                  keyword_list_file:str=None, # can be inferred by input_package_dir
                  keyword_list:list=[], # [KEYWORD1,KEYWORD2,...] <- must be ordered
                  cent_methods:list=['betweenness_centrality'],
                  conn_methods:list=['all_pairs_dijkstra'],
                  cooc_methods:list=['inv_cooccurrence'], # cooccurrence or inv_cooccurrence
                  align_node_order:bool=False,
                  central_node:str='blockchain', # Master keyword to align nodes in the keyword_list
                 ):
    time_specific_dir = os.path.abspath(
        os.path.join(input_package_dir,'time_speicific_graphs'))
    o_d = os.path.abspath(output_dir)
    os.makedirs(o_d,exist_ok=True)
    
    if not whole_time_graph_file:
        whole_time_graph_file = os.path.join(input_package_dir,'combined_graph.pkl')
    wt_G = net_utils.load_graph(whole_time_graph_file)
    
    if not keyword_list:
        with open(keyword_list_file,'rb') as f:
            keyword_list = f.read().decode().split()
    
    # Realign keyword list
    if align_node_order:
        whole_coword_map = {_edg:wt_G.edges[_edg]['cooccurrence:whole_time'] for _edg in wt_G.edges if 'cooccurrence:whole_time' in wt_G.edges[_edg]}
        keyword_list = _realign_keyword_(
            whole_coword_map=whole_coword_map,
            central_node=central_node,
            keyword_list=keyword_list)
    
    # node_indices.tsv
    net_utils._write_node_index_(
        node_ser=pd.Series(keyword_list,index=list(range(len(keyword_list)))),
        output=os.path.join(o_d,'node_indicies.tsv'))
    # edge_list.tsv
    net_utils._write_edge_index_(
        node_list=keyword_list,
        output=os.path.join(o_d,'edge_list.tsv'))
    
    ######## Word count ########
    print('Extracting word count data...')
    # node_targets.npy - word count
    word_count_data = np.array([[wt_G.nodes[kw][f'word_count:{t_k}'] for kw in keyword_list] for t_k in time_key_list])
    net_utils._write_node_features_(
        node_features=word_count_data,
        file=os.path.join(o_d,'word_count.node_targets.npy'),
        gcn_io_format=True)
    # node_attributes.txt - word_count
    with open(os.path.join(o_d,'word_count.node_attributes.txt'),'wb') as f:
        f.write('\n'.join([f'word_count:{_t_k}' for _t_k in time_key_list]).encode())
    
    ######## Cooccurrence ########
    print('Extracting cooccurrence data...')
    # edge_indices.json and edge_weights.json - cooccurrence
    cooccurrence_data = {} #{ATTRB:{NODE1:{NODE2:VALUE}}}
    target_edges = [edge for edge in itertools.permutations(keyword_list,2) if edge in wt_G.edges]
    for _t_k in time_key_list:
        _attrb = 'cooccurrence:'+_t_k
#         _curr_edge_features = {}
        _curr_edge_features = {i:{} for i in keyword_list}
        for edge in target_edges:
            if _attrb in wt_G.edges[edge]:
                n1=edge[0]
                n2=edge[1]
                val = wt_G.edges[edge][_attrb]
#                 if n1 not in _curr_edge_features:
#                     _curr_edge_features[n1] = {}
                _curr_edge_features[n1][n2]=val
    
        cooccurrence_data[_attrb] = net_utils._extr_kw_at_edge_(
            keyword_list=keyword_list,
            edge_features=_curr_edge_features,
            imputation=None,
        )

    _, mapped_cooccurrence_data = net_utils.convert_feature_dicts(
        keyword_list=keyword_list,
        node_feature_dict={},
        edge_feature_dict=cooccurrence_data)
    net_utils._write_edge_features_(
        edge_features=mapped_cooccurrence_data, # {ATTRIBUTE:{SOURCE_NODE:{TARGET_NODE:VALUE}}}
        output_prefix=os.path.join(o_d,'cooccurrence'),
        gcn_io_format=True
    )
        
    # edge_attributes.txt - cooccurrence
    with open(os.path.join(o_d,'cooccurrence.edge_attributes.txt'),'wb') as f:
        f.write('\n'.join(['cooccurrence:'+_t_k for _t_k in time_key_list]).encode())
    
    # Parsing time-specific sub graphs
    print('Extracting centrality and connectivity data...')
    for cooc_method in cooc_methods:
        cent_dict = {cent_method:{} for cent_method in cent_methods} # {METHOD:{ATTRIBUTE(TIME):{NODE:VAL}}}
        conn_dict = {conn_method:{} for conn_method in conn_methods} # {METHOD:{ATTRIBUTE(TIME):{NODE1:{NODE2:VAL}}}}
        for t_k in time_key_list:
            with open(os.path.join(time_specific_dir,f'{t_k}.{cooc_method}.centrality.json'),'rb') as f:
                _curr_cent_dict = json.loads(f.read().decode())
            with open(os.path.join(time_specific_dir,f'{t_k}.{cooc_method}.connectivity.json'),'rb') as f:
                _curr_conn_dict = json.loads(f.read().decode())
            # Extracting values - centrality
            for cent_method in cent_methods:
                _curr_cent_method_dict = _curr_cent_dict[cent_method][cooc_method]
                _curr_cent_method_dict = net_utils._extr_kw_at_node_(
                    keyword_list=keyword_list,
                    node_features=_curr_cent_method_dict,
                    imputation=0.0)
                cent_dict[cent_method][cent_method+':'+t_k] = _curr_cent_method_dict.copy()
            # Extracting values - connectivity
            for conn_method in conn_methods:
                _curr_conn_method_dict =_curr_conn_dict[conn_method][cooc_method]
                _curr_conn_method_dict = net_utils._extr_kw_at_edge_(
                    keyword_list=keyword_list,
                    edge_features=_curr_conn_method_dict,
                    imputation=None)
                conn_dict[conn_method][conn_method+':'+t_k] = _curr_conn_method_dict.copy()
                
        for cent_method in cent_methods:
            # ID mapping
            curr_cent_dict, _ = net_utils.convert_feature_dicts(
                keyword_list=keyword_list,
                node_feature_dict=cent_dict[cent_method],
                edge_feature_dict={})

            # node_targets.npy - centrality
            cent_arr = np.array(
                pd.DataFrame(
                    curr_cent_dict,
                    index=list(range(len(keyword_list))),
                    columns=[cent_method+':'+t_k for t_k in time_key_list],
                ).T)
            net_utils._write_node_features_(
                node_features=cent_arr,
                file=os.path.join(o_d,f'{cent_method}.{cooc_method}.node_targets.npy'),
                gcn_io_format=True,
            )
            # node_attributes.txt - centrality
            with open(os.path.join(o_d,f'{cent_method}.{cooc_method}.node_attributes.txt'),'wb') as f:
                f.write('\n'.join([cent_method+':'+t_k for t_k in time_key_list]).encode())
            
        for conn_method in conn_methods:
            # ID mapping
            _, curr_conn_dict = net_utils.convert_feature_dicts(
                keyword_list=keyword_list,
                node_feature_dict=_,
                edge_feature_dict=conn_dict[conn_method])
            # edge_indices.json and edge_weights.json - connectivity
            nr_conn_dict = {_attrb:net_utils._make_nonredundancy_for_edges_(
                edge_features=_conn_d) for _attrb,_conn_d in curr_conn_dict.items()}

            net_utils._write_edge_features_(
                edge_features=nr_conn_dict,
                output_prefix=os.path.join(o_d,f'{conn_method}.{cooc_method}'),
                gcn_io_format=True,
            )

            # edge_attributes.txt - connectivity
            with open(os.path.join(o_d,f'{conn_method}.{cooc_method}.edge_attributes.txt'),'wb') as f:
                f.write('\n'.join([conn_method+':'+_t_k for _t_k in time_key_list]).encode())
