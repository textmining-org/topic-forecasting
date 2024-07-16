NET_DIR='./../network_analysis/'
TOPIC_DIR='./../analysate_by_2023data/topics/'
OUTPUT_DIR='./../data/patents/'

# Drawing for each topic
python ${NET_DIR}graph_drawing.py \
    -m ${OUTPUT_DIR}3.graph/combined_graph.pkl \
    -o ${OUTPUT_DIR}4.topic.net_figures \
    -t ${TOPIC_DIR}patent_topic_clustering.csv \
    -g ${OUTPUT_DIR}3.graph/tmp_graph_dir \
    --time_line ${OUTPUT_DIR}3.graph/time_lines.txt \
    --node_feature word_count \
    --edge_feature inv_cooccurrence \
    --inverse_edge_width \
    -p ${OUTPUT_DIR}4.topic.net_figures/standard_graph_position.json
    
# Drawing whole graph with label
python ${NET_DIR}graph_drawing.py \
    -m ${OUTPUT_DIR}3.graph/combined_graph.pkl \
    -o ${OUTPUT_DIR}4.topic.net_figures \
    -t ${TOPIC_DIR}patent_topic_clustering.csv \
    -g ${OUTPUT_DIR}3.graph/combined_graph.pkl \
    --time_line ${OUTPUT_DIR}3.graph/time_lines.txt \
    --node_feature word_count:whole_time \
    --edge_feature cooccurrence:whole_time \
    -p ${OUTPUT_DIR}4.topic.net_figures/standard_graph_position.json
    
