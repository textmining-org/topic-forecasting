import os
import networkx as nx
import matplotlib.pyplot as plt
import network_utils as net_utils


# Draw graph - template
def _draw_graph_(G,
                 output:str='output.png',
                 weight:str='weight_1',
                 pos:'networkx.Layout'=None,
                 nodesize_key:str='word_count',
                 edgewidth_key:str='cooccurrence',
                 cmap=plt.cm.RdYlBu_r,
                ):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if not pos:
        pos = nx.kamada_kawai_layout(G,weight=weight)

    nodesize = [G.nodes[node][nodesize_key]*100 for node in G]

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=nodesize,
        node_color=nodesize,
        cmap=cmap,
    )
    nx.draw_networkx_labels(
        G,
        pos=pos,
        font_size=20,
    )

    edgewidth = [G.get_edge_data(u,v)[edgewidth_key]*5 for u, v in G.edges]
    nx.draw_networkx_edges(G,pos=pos,
                           width=edgewidth,
                           edge_color=edgewidth,
                           edge_cmap=cmap,
                           alpha=0.7,
                          )
    fig.tight_layout()
    plt.axis("off")
    plt.savefig(output)
    plt.show()
    return output


def draw_time_serial_graph(graph_loc_d:dict,
                           output:str='output.png',
                           weight:str='weight_1',
                           standard_position:bool=True,
                           standard_position_G_file:str=False,
                           nodesize_key:str='word_count',
                           edgewidth_key:str='cooccurrence',
                           cmap=plt.cm.RdYlBu_r,
                          ):
    output = os.path.abspath(output)
    os.makedirs(output,exist_ok=True)
    # Position standardization for all the time-specific graphs
    if standard_position:
        # Imposed standard position G
        if standard_position_G_file:
            pos_G = net_utils.load_graph(standard_position_G_file)
            
        # Standard at the last graph in ordered graph_loc_d's file location
        else:
            pos_G = net_utils.load_graph(list(graph_loc_d.values())[-1])
        pos = nx.kamada_kawai_layout(pos_G,weight=weight)
        del pos_G
    else:
        pos = None
        
    # position inferrecen
    for g_id, g_file in graph_loc_d.items():
        curr_G = net_utils.load_graph(g_file)
        output = os.path.join(output,g_id+'.graph.png')
        _draw_graph_(
            G=curr_G,
            output=curr_output,
            weight=weight,
            pos=pos,
            nodesize_key=nodesize_key,
            edgewidth_key=edgewidth_key,
            cmap=cmap,
        )
    return output


#### example ####
if __name__=='__main__':
#     G = net_utils.load_graph('./topic-forecasting/output/papers_co10/3.graph/combined_graph.pkl')
#     draw_graph(G=G,output='./papers.combined_graph.whole_time.png',
#         weight='word_count:whole_time',
#         nodesize_key='word_count:whole_time',
#         edgewidth_key='cooccurrence:whole_time',)
    
    G = net_utils.load_graph('./topic-forecasting/output/patents_co10/3.graph/combined_graph.pkl')
    _draw_graph_(
        G=G,
        output=
#         output='./patents.combined_graph.whole_time.png',
        weight='word_count:whole_time',
        nodesize_key='word_count:whole_time',
        edgewidth_key='cooccurrence:whole_time',)
