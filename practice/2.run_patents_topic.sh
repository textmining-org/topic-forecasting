./../network_analysis/run_coword_analysis.py extract_topic \
    -i ./patents_co10/3.graph/ \
    -o ./patents_co10/4.topic/ \
    -t ./patents_co10/3.graph/time_line.txt \
    -k ./patents_topic_clustering.csv \
    --max_keyword_n 50 \
    --cooccurrence inv_cooccurrence \
    --align_node_order \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \

