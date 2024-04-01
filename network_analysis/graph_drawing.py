#!/usr/bin/env python3
"""Modules to draw networkx graphs"""
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import network_utils as net_utils
import argparse
import numpy as np

# Draw graph - template
def _draw_graph_(G,
                 output:str='output.png',
                 weight:str='inv_cooccurrence',
                 pos:'networkx.Layout'=None,
                 node_feature:str='word_count',
                 nodesize_val:float=None,
                 node_alpha:float=1.0,
                 node_cmap=plt.cm.autumn_r,
                 edge_feature:str='cooccurrence',
                 edgewidth_val:float=None,
                 edge_cmap=plt.cm.autumn_r,
                 show_label:bool=True,
                 sub_G_node_list:list=None, # this can be topic
                 with_null_nodes:bool=True,
                 label_options:dict={'font_size':20},
                 edge_options:dict={},
                 edge_width_multiply:float=0.1,
                 edge_width_max_scale:float=False,
                 inverse_edge_width=False,
                ):
    fig, ax = plt.subplots(figsize=(12, 12))
        
    if sub_G_node_list:
        sub_G = G.subgraph(sub_G_node_list).copy()
        if with_null_nodes:
            _new_nodes = list(set(sub_G_node_list)-set(sub_G.nodes))
            sub_G.add_nodes_from([(n,{node_feature:0.0}) for n in _new_nodes])
    else:
        sub_G = G
    
    if not pos:
        print(f"Getting position for graph object: \"{weight}\" as weight")
        _pos = nx.kamada_kawai_layout(sub_G,weight=weight)
    else:
        _pos = {_k:_v for _k,_v in pos.items() if _k in sub_G.nodes}
    if nodesize_val:
        nodesize = [nodesize_val for node in sub_G.nodes]
    else:
        nodesize = [sub_G.nodes[node][node_feature]*100 for node in sub_G.nodes]
    if node_cmap:
        nodecolor = [sub_G.nodes[node][node_feature]*100 for node in sub_G.nodes]
    else:
        nodecolor = [1.0 for node in sub_G.nodes]
    nx.draw_networkx_nodes(
        sub_G,
        pos=_pos,
        node_size=nodesize,
        node_color=nodecolor,
        cmap=node_cmap,
        alpha=node_alpha,
    )
    if show_label:
        nx.draw_networkx_labels(
            sub_G,
            pos=_pos,
            **label_options,
        )
    if sub_G.edges:
        if edgewidth_val:
            edgewidth = [edgewidth_val for u, v in sub_G.edges]
        else:
            edgewidth = [sub_G.get_edge_data(u,v)[edge_feature] for u, v in sub_G.edges]
        if inverse_edge_width:
            edgewidth = [1.0/i*edge_width_multiply for i in edgewidth]
        else:
            edgewidth = [i*edge_width_multiply for i in edgewidth]
        if edge_width_max_scale:
            edgewidth = list(np.array(edgewidth)/np.max(edgewidth)*edge_width_max_scale)
        if edge_cmap:
            edgecolor = [sub_G.get_edge_data(u,v)[edge_feature]*edge_width_multiply for u, v in sub_G.edges]
        else:
            edgecolor = [1.0 for u, v in sub_G.edges]
        if inverse_edge_width:
            edgecolor = [1./i for i in edgecolor]
        nx.draw_networkx_edges(sub_G,
                               pos=_pos,
                               width=edgewidth,
                               edge_color=edgecolor,
                               edge_cmap=edge_cmap,
                               **edge_options,
    #                            # TODO REMOVE HERE
    #                            edge_color="r",
    #                            alpha=0.7,
    #                            # BY HERE
                          )
    fig.tight_layout()
    plt.axis("off")
    if output:
        plt.savefig(output)
    plt.show()
    return output


def draw_time_serial_graph(graph_loc_d:dict,
                           output:str='output',
                           pos_weight:str='inv_cooccurrence',
                           standard_position:bool=True,
                           standard_position_file:str=False,
                           standard_position_G_file:str=False,
                           node_feature:str='word_count',
                           nodesize_val:float=None,
                           node_alpha:float=1.0,
                           node_cmap=plt.cm.autumn_r,
                           edge_feature:str='cooccurrence',
                           edgewidth_val:float=None,
                           edge_cmap=plt.cm.autumn_r,
                           edge_width_multiply=0.1,
                           edge_width_max_scale=False,
                           edge_options:dict={},
                           save_pos=False,
                           sub_G_node_list:list=[],
                           show_label=True,
                           inverse_edge_width=True,
                          ):
    output = os.path.abspath(output)
    os.makedirs(output,exist_ok=True)
    # Position standardization for all the time-specific graphs
    if standard_position:
        # Imposed standard position G
        if standard_position_file:
            print("Parsing standard position file...")
            pos = net_utils._load_graph_position_(standard_position_file)
        elif standard_position_G_file:
            print("Parsing graph file...")
            pos_G = net_utils.load_graph(standard_position_G_file)
            pos_subg = pos_G.subgraph(sub_G_node_list)
            print("Calculating standard position...")
            pos = nx.kamada_kawai_layout(pos_subg,weight=pos_weight)
            del pos_subg
        # Standard at the last graph in ordered graph_loc_d's file location
        else:
            print("Parsing graph file %s for standard position..."%(list(graph_loc_d.values())[-1]))
            pos_G = net_utils.load_graph(list(graph_loc_d.values())[-1])
            print("Calculating standard position...")
            pos = nx.kamada_kawai_layout(pos_G,weight=pos_weight)
            del pos_G
    else:
        pos = None
    if save_pos:
        net_utils._save_graph_position_(
            pos=pos,file=os.path.join(output,'standard_graph_position.json'))
        print("Standard position has been saved to %s"%os.path.join(
            output,'standard_graph_position.json'))
    # position inferrecen
    for g_id, g_file in graph_loc_d.items():
        print("Generating graph for %s : %s"%(g_id,g_file))
        curr_G = net_utils.load_graph(g_file)
#         curr_output = os.path.join(output,g_id+'.graph.png')
        curr_output = os.path.join(output,g_id+'.graph.svg')
        if sub_G_node_list:
            curr_target_nodes = sub_G_node_list
        else:
            curr_target_nodes = list(curr_G.nodes)
        _draw_graph_(
            G=curr_G,
            output=curr_output,
            weight=pos_weight,
            pos=pos,
            node_feature=node_feature,
            nodesize_val=nodesize_val,
            node_alpha=node_alpha,
            node_cmap=node_cmap,
            edge_feature=edge_feature,
            edgewidth_val=edgewidth_val,
            edge_cmap=edge_cmap,
            edge_options=edge_options,
            sub_G_node_list=curr_target_nodes,
            edge_width_multiply=edge_width_multiply,
            show_label=show_label,
            inverse_edge_width=inverse_edge_width,
            edge_width_max_scale=edge_width_max_scale,
        )
    return output


#### example ####

def main():
    parser = argparse.ArgumentParser(description='Graph drawing by time')
    parser.add_argument('-m','--master_graph',help='Input master graph file')
    parser.add_argument('-g','--time_graph',default=None,help='Time-specific graph directory')
    parser.add_argument('-t','--topic_file',help='Topic file with keywords')
    parser.add_argument('-o','--output',help='Output directory')
    parser.add_argument('--time_line',default=None,help='Timeline file')
    parser.add_argument('-p','--pos_file',default='',help='Position file')
    parser.add_argument('--node_feature',default='word_count',help='Feature for node size and color')
    parser.add_argument('--edge_feature',default='cooccurrence',help='Feature for edge size and color')
    parser.add_argument('--inverse_edge_width',default=False,action='store_true',help='Inverse value of edge feature')
    
    args = parser.parse_args()
    
    pos = None
    os.makedirs(args.output,exist_ok=True)
    # Single
    if args.time_line:
        time_file=args.time_line
        times = pd.read_csv(time_file,header=None).iloc[:,0].values
    else:
        times = {}
    if os.path.isfile(args.time_graph):
        label_graph_d = {'label_graph':args.time_graph}
    if os.path.isdir(args.time_graph):
        annod_d = {t:os.path.join(args.time_graph,f'{t}.inv_cooccurrence.graph.pkl') for t in times}
    pos_file=args.pos_file
    # Standard at the last graph in ordered graph_loc_d's file location
    if not os.path.isfile(pos_file):
        pos_file = os.path.join(args.output,'standard_graph_position.json')
        print("Parsing graph file %s for standard position..."%(args.master_graph))
        # getting graph position
        pos_G = net_utils.load_graph(args.master_graph)
        print("Calculating standard position...")
        pos_weight='inv_cooccurrence:whole_time',
        pos = nx.kamada_kawai_layout(pos_G,weight=pos_weight)
        del pos_G
        net_utils._save_graph_position_(
            pos=pos,file=pos_file)
        print("Standard position has been saved to %s"%os.path.join(
            args.output,'standard_graph_position.json'))
        del pos
        
    if args.topic_file:
        topic_df = pd.read_csv(args.topic_file)
        for idx in topic_df.index:
            [topic_id, kw_str] = topic_df.loc[idx,:].values
            _kws = list(set(kw_str.split(' ')))
            curr_output = os.path.join(args.output,f'topic_graph.{topic_id}')
            # without label
            if os.path.isdir(args.time_graph):
                draw_time_serial_graph(
                    graph_loc_d=annod_d,
                    output=curr_output,
                    pos_weight='inv_cooccurrence',
                    standard_position=True,
                    standard_position_G_file=args.master_graph,
                    standard_position_file=pos_file,
                    node_feature=args.node_feature, # 'word_count'
                    nodesize_val=1000., # constant node size
                    node_alpha=0.7,
                    node_cmap=plt.cm.magma_r, # color map
                    edge_feature=args.edge_feature, #'inv_cooccurrence',
#                     edgewidth_val=1, # constant edge width
                    edge_cmap=plt.cm.magma_r, # color map
                    edge_width_multiply=1,
                    edge_width_max_scale=5,
                    save_pos=False,#=True
                    show_label=False,
                    sub_G_node_list=_kws,
                    inverse_edge_width=args.inverse_edge_width,
                )
            # label-only graph
            if os.path.isfile(args.time_graph):
                draw_time_serial_graph(
                    graph_loc_d=label_graph_d,
                    output=curr_output,
                    pos_weight='inv_cooccurrence',
                    standard_position=True,
                    standard_position_G_file=args.master_graph,
                    standard_position_file=pos_file,
                    node_feature=args.node_feature, # 'word_count:whole_time'
                    nodesize_val=1000.0, # constant node size
                    node_cmap=None, # color map
                    node_alpha=0.2,
                    edge_feature=args.edge_feature, # 'cooccurrence:whole_time'
                    edgewidth_val=1, # constant edge width
                    edge_cmap=None,
                    edge_width_multiply=1,
                    edge_width_max_scale=5,
                    save_pos=False,#=True
                    show_label=True,
                    sub_G_node_list=_kws,
                    edge_options={'alpha':0.2},
                    inverse_edge_width=args.inverse_edge_width,
                )
#                 draw_time_serial_graph(
#                     graph_loc_d=label_graph_d,
#                     output=curr_output,
#                     pos_weight='inv_cooccurrence',
#                     standard_position=True,
#                     standard_position_G_file=args.master_graph,
#                     standard_position_file=pos_file,
#                     node_feature=args.node_feature, # 'word_count:whole_time'
#                     nodesize_val=1000.0, # constant node size
#                     node_cmap=None, # color map
#                     node_alpha=0.2,
#                     edge_feature=args.edge_feature, # 'cooccurrence:whole_time'
#                     edgewidth_val=1, # constant edge width
#                     edge_cmap=None,
#                     edge_width_multiply=1,
#                     edge_width_max_scale=5,
#                     save_pos=False,#=True
#                     show_label=False,
#                     sub_G_node_list=_kws,
#                     edge_options={'alpha':0.2},
#                     inverse_edge_width=args.inverse_edge_width,
#                 )
    else:
        draw_time_serial_graph(
            graph_loc_d=annod_d,
            output=args.output,
            pos_weight='cooccurrence',
            standard_position=True,
            standard_position_G_file=args.master_graph,#'./topic-forecasting/output/patents_co10/3.graph/combined_graph.pkl',
            standard_position_file=pos_file,
            node_feature=args.node_feature,
            nodesize_val=1.0, # constant node size
            edge_feature=args.edge_feature,
            edgewidth_val=1.0, # constant edge width
            node_cmap=plt.cm.magma_r,
            edge_cmap=plt.cm.magma_r,
            edge_width_multiply=0.1,
            show_label=False,
            save_pos=False,#=True
            inverse_edge_width=args.inverse_edge_width,
        )
        
        
if __name__=='__main__':
    main()
              