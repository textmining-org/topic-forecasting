import os
import networkx as nx
import matplotlib.pyplot as plt
import network_utils as net_utils


# Draw graph - template
def _draw_graph_(G,
                 output:str='output.png',
                 weight:str='inv_cooccurrence',
                 pos:'networkx.Layout'=None,
                 nodesize_key:str='word_count',
                 edgewidth_key:str='cooccurrence',
                 cmap=plt.cm.RdYlBu_r,
                 show_label:bool=True,
                 sub_G_node_list:list=None,
                 label_options:dict={'font_size':20},
                 edge_options:dict={'alpha':0.7},
                 edge_width_multiply:float=5.,
                ):
    fig, ax = plt.subplots(figsize=(12, 12))
        
    if sub_G_node_list:
        sub_G = G.subgraph(sub_G_node_list)
    else:
        sub_G = G
    
    if not pos:
        print(f"Getting position for graph object: \"{weight}\" as weight")
        _pos = nx.kamada_kawai_layout(sub_G,weight=weight)
    else:
        _pos = {_k:_v for _k,_v in pos.items() if _k in sub_G.nodes}

    nodesize = [sub_G.nodes[node][nodesize_key]*100 for node in sub_G.nodes]
    
    nx.draw_networkx_nodes(
        sub_G,
        pos=_pos,
        node_size=nodesize,
        node_color=nodesize,
        cmap=cmap,
    )
    if show_label:
        nx.draw_networkx_labels(
            sub_G,
            pos=_pos,
            **label_options,
        )
    edgewidth = [sub_G.get_edge_data(u,v)[edgewidth_key]*edge_width_multiply for u, v in sub_G.edges]
    nx.draw_networkx_edges(sub_G,
                           pos=_pos,
                           width=edgewidth,
                           edge_color=edgewidth,
                           edge_cmap=cmap,
                           **edge_options,
#                            # TODO REMOVE HERE
#                            width=1.0,
#                            edge_color="r",
#                            edge_cmap=None,
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
                           output:str='output_dir',
                           pos_weight:str='inv_cooccurrence',
                           standard_position:bool=True,
                           standard_position_file:str=False,
                           standard_position_G_file:str=False,
                           nodesize_key:str='word_count',
                           edgewidth_key:str='cooccurrence',
                           cmap=plt.cm.RdYlBu_r,
                           edge_width_multiply=5.,
                           save_pos=False,
                          ):
    output = os.path.abspath(output)
    os.makedirs(output,exist_ok=True)
    # Position standardization for all the time-specific graphs
    if standard_position:
        # Imposed standard position G
        if standard_position_file:
            pos = net_utils._load_graph_position_(standard_position_file)
        elif standard_position_G_file:
            pos_G = net_utils.load_graph(standard_position_G_file)
            pos = nx.kamada_kawai_layout(pos_G,weight=pos_weight)
            del pos_G
        # Standard at the last graph in ordered graph_loc_d's file location
        else:
            pos_G = net_utils.load_graph(list(graph_loc_d.values())[-1])
            pos = nx.kamada_kawai_layout(pos_G,weight=pos_weight)
            del pos_G
    else:
        pos = None
    if save_pos:
        net_utils._save_graph_position_(
            pos=pos,file=os.path.join(output_dir,'standard_graph_position.json'))
    # position inferrecen
    for g_id, g_file in graph_loc_d.items():
        curr_G = net_utils.load_graph(g_file)
        output = os.path.join(output,g_id+'.graph.png')
        _draw_graph_(
            G=curr_G,
            output=curr_output,
            weight=pos_weight,
            pos=pos,
            nodesize_key=nodesize_key,
            edgewidth_key=edgewidth_key,
            cmap=cmap,
            sub_G_node_list=list(curr_G.nodes),
        )
    return output


#### example ####
if __name__=='__main__':
#     G = net_utils.load_graph('./topic-forecasting/output/papers_co10/3.graph/combined_graph.pkl')
#     draw_graph(G=G,output='./papers.combined_graph.whole_time.png',
#         weight='word_count:whole_time',
#         nodesize_key='word_count:whole_time',
#         edgewidth_key='cooccurrence:whole_time',)
    
    pos = None
    pos_file = None
    # Single
#     G = net_utils.load_graph('./topic-forecasting/output/patents_co10/3.graph/combined_graph.pkl')
#     _draw_graph_(
#         G=G,
#         output='./test_result.graph.plot.png',
# #         output='./patents.combined_graph.whole_time.png',
#         pos_weight='inv_cooccurrence:whole_time',
#         nodesize_key='word_count:whole_time',
#         edgewidth_key='cooccurrence:whole_time',)

    
    time_file = './topic-forecasting/output/time_line.txt' # annotation
    times = pd.read_csv('./topic-forecasting/output/time_line.txt',header=None).iloc[:,0].values
    annod_d = {t:f'./topic-forecasting/output/patents_co10/3.graph/time_speicific_graphs/{t}.cooccurrence.graph.pkl' for t in times}
    draw_time_serial_graph(
        graph_loc_d=annod_d,
        output=f'./topic-forecasting/output/patents_co10_test_graphs',
        pos_weight='cooccurrence',
        standard_position=True,
        standard_position_G_file='./topic-forecasting/output/patents_co10/3.graph/combined_graph.pkl',
        standard_position_file=pos_file,
        nodesize_key='word_count',
        edgewidth_key='cooccurrence',
        cmap=plt.cm.RdYlBu_r,
        edge_width_multiply=0.1,
        save_pos=True,
    )