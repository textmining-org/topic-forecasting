NET_DIR='./../network_analysis/'
PREPROCESSED_DATA='./../analysate_by_2023data/_preprocessed_dataset_/patent_201701_202312_without_duplicates.pkl'
OUTPUT_DIR='./../data/patents/'

# Coword detection
python ${NET_DIR}run_coword_analysis.py get_coword \
    -i ${PREPROCESSED_DATA} \
    -o ${OUTPUT_DIR}2.coword \
    --word_count_limit 10 \

# Make graph for coword mapping file
python ${NET_DIR}run_coword_analysis.py make_graph \
    -i ${OUTPUT_DIR}2.coword/coword_results.pkl \
    -o ${OUTPUT_DIR}3.graph \
    -m 8 \
    --centrality degree_centrality \
    --centrality betweenness_centrality \
    --centrality closeness_centrality \

