
# # Draw graph - template
# def draw_graph(G,
#                output:str,
#                weight:str='weight_1',
#                layout_weight:str='weight_1',
#                nodesize_key:str,
#                edgewidth_key:str):
#     fig, ax = plt.subplots(figsize=(12, 12))

#     pos = nx.kamada_kawai_layout(G,weight='weight_1')

#     nodesize_key = 'frequency'
#     nodesize = [G.nodes[node][nodesize_key]*100 for node in G]

#     nx.draw_networkx_nodes(
#         G,
#         pos=pos,
#         node_size=nodesize,
#         node_color=nodesize,
#         cmap=plt.cm.RdYlBu_r,
#     )
#     nx.draw_networkx_labels(
#         G,
#         pos=pos,
#         font_size=20,
#     )

#     edgewidth_key = 'weight'
#     edgewidth = [G.get_edge_data(u,v)[edgewidth_key]*5 for u, v in G.edges]
#     nx.draw_networkx_edges(G,pos=pos,
#                            width=edgewidth,
#                            edge_color=edgewidth,
#                            edge_cmap=plt.cm.RdYlBu_r,
#                            alpha=0.7,
#                           )

#     fig.tight_layout()
#     plt.axis("off")
#     plt.savefig(output)
#     plt.show()
#     return output