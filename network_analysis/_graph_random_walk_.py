#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import network_utils as net_utils
import copy
import _multi


# SEED = 123
# np.random.seed(SEED)

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


# weight_fn : function of probability over word count. e.g. y=x, y=e^x
# default is y=x
def _pick_weight_random_node_(graph_obj,source_node_list:list,weight_fn=None):
    _s_t_d = _get_edge_d_(graph_obj,node_list=source_node_list)
    _ts = []
    for _s, _t in _s_t_d.items():
        _ts.extend(list(set(_t)-set(source_node_list)))
    if _ts:
        if weight_fn == None:
            return _pick_random_node_(node_list=_ts)
        else:
            _t_count_d = {wrd:_ts.count(wrd) for wrd in set(_ts)}
            p_dict = {wrd:weight_fn(cnt) for wrd, cnt in _t_count_d.items()}
            return np.random.choice(list(p_dict.keys()),1,p=list(p_dict.values()))[0]
    # The cluster is island
    else:
        return None


# Impose argument (graph_obj) or (node_list and edge_dict)
# edge_dict (dict): {NODE1:[NODE2]}
# seed_nodes (list, optional): list of nodes to start random walking
def make_random_node_cluster(graph_obj,
                             seed_node_pool_list=[], # To specify node pool
                             seed_nodes:list=[],
                             node_n:int=None, # target node number
                             min_node_n:int=8,
                             max_node_n:int=20,
                             weight_fn=None,
                            ):
    if not seed_node_pool_list:
        _n_l = _get_node_l_(graph_obj)
    else:
        _n_l = seed_node_pool_list.copy()
        
    if not seed_nodes:
        curr_node_pool = [_pick_random_node_(node_list=_n_l)]
    else:
        curr_node_pool = seed_nodes.copy()
    if node_n:
        iter_node_n = node_n
    else:
        iter_node_n = np.random.randint(min_node_n,max_node_n+1)
#     if weight_fn == None:
#         weight_fn = lambda x: x
    while len(curr_node_pool)<iter_node_n:
        new_node = _pick_weight_random_node_(
            graph_obj=graph_obj,
            source_node_list=curr_node_pool,
            weight_fn=None)
        if not new_node:
            break
        elif new_node in curr_node_pool:
            continue
        else:
            curr_node_pool.append(new_node)
#         _s = _pick_random_node_(node_list=curr_node_pool) # pick random source node from seed list
#         _t_list = list(set(_e_d[_s])-set(curr_node_pool)) # get targets from edge dictionary
#         if _t_list:
#             curr_node_pool.append(_pick_random_node_(node_list=_t_list))
    return curr_node_pool


def remove_low_rated_edge(graph_obj,
                          edge_drop_thr:float=None,
                          _lower_limit_:float=1.0,
                          ):
    g_copy = graph_obj.copy()
    if not edge_drop_thr:
        drop_limit = _lower_limit_
    elif edge_drop_thr<_lower_limit_:
        drop_limit = _lower_limit_
    else:
        drop_limit = edge_drop_thr
    for _e in graph_obj.edges:
        if len(graph_obj.edges[_e])<=drop_limit: # At least more than one edges should be exist - cooccurrence:whole_time
            g_copy.remove_edges_from([_e])
    return g_copy


# cluster_dict = {CLUSTER_ID:[KWRD]}
def get_unique_kwrd_sets(cluster_dict:dict):
    cluster_kwrd_sets = set([tuple(sorted(list(set(i)))) for i in cluster_dict.values()])
    dropped_kwrd_sets = [list(i) for i in cluster_kwrd_sets]
    return dropped_kwrd_sets


def make_random_clusters(graph_obj,
                         seed_node_list:list=[], # Seed nodes to start clustering
                         cluster_min:int=0,
                         cluster_max:int=100,
                         min_node_n:int=8,
                         max_node_n:int=20,
                         tmp_output_file:str='tmp.random_batch.0.json',
                         random_seed_int:int=123,
                        ):
    np.random.seed(random_seed_int)
    cluster_dict = {} # {CLUSTER_NAME:[KEYWORDS]}
    if type(graph_obj) == str:
        graph_obj = net_utils.load_graph(graph_obj)
#     _n_l = _get_node_l_(graph_obj)
    _e_d = _get_edge_d_(graph_obj)
    _n_l = [s for s,t in _e_d.items() if len(t)>=1]
    print(tmp_output_file+':'+str(cluster_max-cluster_min))
    c = 0
    for clstr_no in range(cluster_min,cluster_max):
        c += 1
        cluster_name = f'random_cluster_{clstr_no:04d}'
        kwrds = []
        while len(kwrds)<min_node_n:
            kwrds = make_random_node_cluster(
                graph_obj,
                seed_node_pool_list=_n_l,
                seed_nodes=seed_node_list,
                node_n=None,
                min_node_n=min_node_n,
                max_node_n=max_node_n,
            )
        cluster_dict[cluster_name] = kwrds
        if c%20==0:
            print(tmp_output_file+' processed: '+str(c/(cluster_max-cluster_min)))
    print("Random clustering finished for batch %s"%tmp_output_file)
    with open(tmp_output_file,'wb') as f:
        f.write(json.dumps(cluster_dict).encode())
        
    return cluster_dict


def random_cluster(whole_time_graph_file:str,
                   exclusive_node_file:str=None, # \n delimitted file of keywords
                   output_f='./random_clusters.json',
                   seed_node_file:str=None, # \n delimitted file of keywords
                   edge_drop_thr:int=8,
                   cluster:int=100,
                   min_node_n:int=8,
                   max_node_n:int=20,
                   multiprocess:int=8,
                  ):
    graph_obj = net_utils.load_graph(whole_time_graph_file)
    # exclusive node
    if exclusive_node_file:
        excl_n_l = []
        excl_n_d = net_utils.parse_keyword_file(exclusive_node_file)
        for _v in excl_n_d.values():
            excl_n_l.extend(_v)
        excl_n_l = list(set(excl_n_l))
        print(f'{len(excl_n_l)} words are found for exclusive words')
        print(f"Input graph: {len(graph_obj.nodes)} words with {len(graph_obj.edges)}")
        graph_obj.remove_nodes_from(excl_n_l)
        print(f"Exclusive node_excluded graph: {len(graph_obj.nodes)} words with {len(graph_obj.edges)}")
        
    # remove edges lower occurrent edges
    graph_obj = remove_low_rated_edge(
        graph_obj=graph_obj,edge_drop_thr=edge_drop_thr)
        
    net_utils.save_graph(graph_obj=graph_obj,output_file=output_f+'.tmp_graph_obj.pkl')
    del graph_obj
    if seed_node_file:
        with open(seed_node_file,'rb') as f:
            seed_node_list=f.read().decode().split()
    else:
        seed_node_list=[]
    fn_arg_list = []
    fn_kwarg_list = []
    tmp_output_file_list = []
    if cluster<multiprocess:
        multiprocess = 1
    clstr_step = int(cluster/multiprocess)
    for _batch_n in range(multiprocess):
        tmp_output_file = output_f+'.random_batch.%s.json'%_batch_n
        tmp_output_file_list.append(tmp_output_file)
        _curr_kwargs = dict(
            graph_obj=output_f+'.tmp_graph_obj.pkl',
            seed_node_list=seed_node_list,
            cluster_min=int(clstr_step*_batch_n),
            cluster_max=int(clstr_step*(_batch_n+1)),
            min_node_n=min_node_n,
            max_node_n=max_node_n,
            tmp_output_file=tmp_output_file,
            random_seed_int=_batch_n,
        )
        if _batch_n==multiprocess:
            _curr_kwargs['max_node_n'] = cluster
        fn_arg_list.append(tuple())
        fn_kwarg_list.append(copy.deepcopy(_curr_kwargs))
    fn_args = _multi.argument_generator(fn_arg_list)
    fn_kwargs = _multi.keyword_argument_generator(fn_kwarg_list)
    
    result = _multi.multi_function_execution(
        fn=make_random_clusters,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        max_processes=multiprocess,
        collect_result=False,
    )
    
    clstr_dict = {}
    for _f in tmp_output_file_list:
        with open(_f,'rb') as f:
            _curr_cltr_dict = json.loads(f.read().decode())
        clstr_dict.update(_curr_cltr_dict.copy())
        os.system('rm %s'%_f)
        
    unq_kwrds = get_unique_kwrd_sets(clstr_dict)
    unq_clstr_dict = {}
    for clstr_no, kwrd_l in enumerate(unq_kwrds):
        cluster_name = f'random_cluster_{clstr_no:04d}'
        unq_clstr_dict[cluster_name] = kwrd_l
        
    if output_f.endswith('json'):
        with open(output_f,'wb') as f:
            f.write(json.dumps(unq_clstr_dict).encode())
    else:
        clstr_df = pd.DataFrame(
            {_k:{'keywords':' '.join(_kwd_l)} for _k, _kwd_l in unq_clstr_dict.items()}
        ).T
        clstr_df.index.name = 'cluster_name'
        if output_f.endswith('csv'):
            clstr_df.to_csv(clstr_df)
        elif output_f.endswith('tsv'):
            clstr_df.to_csv(clstr_df,sep='\t')
        elif output_f.endswith('pkl'):
            with open(output_f,'wb') as f:
                pickle.dump(clstr_df,f)
    
    