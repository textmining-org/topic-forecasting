# Coword detection
./../network_analysis/run_coword_analysis.py get_coword \
    -i ./_preprocessed_dataset_/until_202212/patents.pkl \
    -o ./patents_co10/2.coword \
    --word_count_limit 10 \

# Make graph for coword mapping file
./../network_analysis/run_coword_analysis.py make_graph \
    -i ./patents_co10/2.coword/coword_results.pkl \
    -o ./patents_co10/3.graph \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \
#    --centrality eigenvector_centrality \ # fail during iteration:  networkx.exception.PowerIterationFailedConvergence: (PowerIterationFailedConvergence(...), 'power iteration failed to converge within 100 iterations')

