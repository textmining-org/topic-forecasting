NET_DIR='./../network_analysis/'
TOPIC_DIR='./../analysate_by_2023data/topics/'
OUTPUT_DIR='./../data/patents/'

mkdir ${OUTPUT_DIR}5.random_cluster.wo_topic

python ${NET_DIR}run_coword_analysis.py random_cluster \
    -i ${OUTPUT_DIR}3.graph/combined_graph.pkl \
    -o ${OUTPUT_DIR}5.random_cluster.wo_topic/random_clusters.json \
    -m 16 \
    --exclusive_node_file ${TOPIC_DIR}patent_topic_clustering.csv \
    --max_node_n 20 \
    --min_node_n 8 \
    --cluster_n 10000 \
    --edge_ens_span 48 \
    --edge_drop_thr 1 \
    -t ${OUTPUT_DIR}3.graph/time_lines.txt \

python ${NET_DIR}run_coword_analysis.py extract_topic \
    -i ${OUTPUT_DIR}3.graph/ \
    -o ${OUTPUT_DIR}5.random_cluster.wo_topic/clusters/ \
    -t ${OUTPUT_DIR}3.graph/time_lines.txt \
    -k ${OUTPUT_DIR}5.random_cluster.wo_topic/random_clusters.json \
    -m 16 \
    --max_keyword_n 20 \
    --cooccurrence inv_cooccurrence \
    --align_node_order \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \
