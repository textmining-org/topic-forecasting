# mkdir ./patents_co10/5.random_cluster

# ./../network_analysis/run_coword_analysis.py random_cluster \
#     -i ./patents_co10/3.graph/combined_graph.pkl \
#     -o ./patents_co10/5.random_cluster/random_clusters.json \
#     --exclusive_node_file ./patents_topic_clustering.csv \
#     --max_node_n 50 \
#     --min_node_n 10 \
#     --cluster_n 10000 \


./../network_analysis/run_coword_analysis.py extract_topic \
    -i ./patents_co10/3.graph/ \
    -o ./patents_co10/5.random_cluster/clusters/ \
    -t ./patents_co10/3.graph/time_line.txt \
    -k ./patents_co10/5.random_cluster/random_clusters.json \
    -m 16 \
    --max_keyword_n 50 \
    --cooccurrence inv_cooccurrence \
    --align_node_order \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \
#     --connectivity all_pairs_node_connectivity \
#     --connectivity all_pairs_dijkstra \
#     --connectivity all_pairs_bellman_ford_path_length \
#     --connectivity edge_betweenness_centrality \

