#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import networkx as nx
import network_utils as net_utils


SEED = 123
np.random.seed(SEED)

def _pick_random_node_(graph_obj=None,node_list=None):
    if graph_obj:
        _n_l = list(graph_obj.nodes)
    elif node_list:
        _n_l = node_list
    else:
        _n_l = []
    return _n_l[np.random.randint(0,len(_n_l))]


def _get_node_l_(graph_obj):
    return list(graph_obj.nodes)


# node_list (list) : (optional) to inclusively select edge dictionary. Only edges with nodes in the node list would be selected.
def _get_edge_d_(graph_obj,node_list:list=[]):
    if node_list:
        _n_l = node_list
    else:
        _n_l = _get_node_l_(graph_obj)
    _e_l = list(graph_obj.edges)
    _pick_rest = lambda _set_tup, x: _set_tup[0] if x==_set_tup[1] else _set_tup[1]
    _e_d = {_n1:[_pick_rest(_edg,_n1) for _edg in _e_l if _n1 in _edg] for _n1 in _n_l}
    return _e_d

# Impose argument (graph_obj) or (node_list and edge_dict)
# edge_dict (dict): {NODE1:[NODE2]}
# seed_nodes (list, optional): list of nodes to start random walking
def make_random_node_cluster(graph_obj,
                             node_pool_list=[], # To specify node pool
                             edge_pool_dict={}, # To specify edge pool
                             seed_nodes:list=[],
                             node_n:int=None, # target node number
                             min_node_n:int=10,
                             max_node_n:int=40,
                            ):
    if not node_pool_list and not edge_pool_dict:
        _n_l = _get_node_l_(graph_obj)
        _e_d = _get_edge_d_(graph_obj)
    else:
        _n_l = node_pool_list.copy()
        _e_d = edge_pool_dict.copy()
        
    if not seed_nodes:
        seed_nodes = [_pick_random_node_(node_list=_n_l)]
    if node_n:
        iter_node_n = node_n
    else:
        iter_node_n = np.random.randint(min_node_n,max_node_n+1)
    while len(seed_nodes)<iter_node_n:
        _s = _pick_random_node_(node_list=seed_nodes) # pick random source node from seed list
        _t_list = list(set(_e_d[_s])-set(seed_nodes)) # get targets from edge dictionary
        if _t_list:
            seed_nodes.append(_pick_random_node_(node_list=_t_list))
    return seed_nodes



def make_random_clusters(graph_obj,
                         seed_node_list:list=[], # Seed nodes to start clustering
                         cluster:int=100,
                         min_node_n:int=10,
                         max_node_n:int=40,
                        ):
    cluster_dict = {} # {CLUSTER_NAME:[KEYWORDS]}
    _n_l = _get_node_l_(graph_obj)
    _e_d = _get_edge_d_(graph_obj)
    
    for clstr_no in range(cluster):
        cluster_name = f'random_cluster_{clstr_no:04d}'
        kwrds = make_random_node_cluster(
            graph_obj,
            node_pool_list=_n_l,
            edge_pool_dict=_e_d,
            seed_nodes=seed_node_list,
            node_n=None,
            min_node_n=min_node_n,
            max_node_n=max_node_n,
        )
        cluster_dict[cluster_name] = kwrds
        
    return cluster_dict


def random_cluster(whole_time_graph_file:str,
                   output_f='./random_clusters.json',
                   seed_node_file:str=None, # \n delimitted file of keywords
                   cluster:int=100,
                   min_node_n:int=10,
                   max_node_n:int=40,):
    graph_obj = net_utils.load_graph(whole_time_graph_file)
    if seed_node_file:
        with open(seed_node_file,'rb') as f:
            seed_node_list=f.read().decode().split()
    else:
        seed_node_list=None
    clstr_dict = make_random_clusters(
        graph_obj=graph_obj,
        seed_node_list=seed_node_list,
        cluster=cluster,
        min_node_n=min_node_n,
        max_node_n=max_node_n,
    )
    if output_f.endswith('json'):
        with open(output_f,'wb') as f:
            f.write(json.dumps(clstr_dict).encode())
    else:
        clstr_df = pd.DataFrame(
            {_k:{'keywords':' '.join(_kwd_l)} for _k, _kwd_l in clstr_dict.items()}
        ).T
        clstr_df.index.name = 'cluster_name'
        if output_f.endswith('csv'):
            clstr_df.to_csv(clstr_df)
        elif output_f.endswith('tsv'):
            clstr_df.to_csv(clstr_df,sep='\t')
        elif output_f.endswith('pkl'):
            with open(output_f,'wb') as f:
                pickle.dump(clstr_df,f)
    
    