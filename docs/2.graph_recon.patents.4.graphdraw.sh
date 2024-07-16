# Drawing for each topic
./../network_analysis/graph_drawing.py \
    -m ./patents_co10/3.graph/combined_graph.pkl \
    -o ./patents_co10/4.topic.net_figures \
    -t ./topics/patent_topic_clustering.csv \
    -g ./patents_co10/3.graph/tmp_graph_dir \
    --time_line ./patents_co10/3.graph/time_lines.txt \
    --node_feature word_count \
    --edge_feature inv_cooccurrence \
    --inverse_edge_width \
    -p ./patents_co10/4.topic.net_figures/standard_graph_position.json
    
# Drawing whole graph with label
./../network_analysis/graph_drawing.py \
    -m ./patents_co10/3.graph/combined_graph.pkl \
    -o ./patents_co10/4.topic.net_figures \
    -t ./topics/patent_topic_clustering.csv \
    -g ./patents_co10/3.graph/combined_graph.pkl \
    --time_line ./patents_co10/3.graph/time_lines.txt \
    --node_feature word_count:whole_time \
    --edge_feature cooccurrence:whole_time \
    -p ./patents_co10/4.topic.net_figures/standard_graph_position.json
    
