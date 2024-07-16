NET_DIR='./../network_analysis/'
TOPIC_DIR='./../analysate_by_2023data/topics/'
OUTPUT_DIR='./../data/patents/'
python ${NET_DIR}run_coword_analysis.py extract_topic \
    -i ${OUTPUT_DIR}3.graph/ \
    -o ${OUTPUT_DIR}4.topic/ \
    -t ${OUTPUT_DIR}3.graph/time_lines.txt \
    -k ${TOPIC_DIR}patent_topic_clustering.csv \
    --max_keyword_n 20 \
    --cooccurrence inv_cooccurrence \
    --align_node_order \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \

