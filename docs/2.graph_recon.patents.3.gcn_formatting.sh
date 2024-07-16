# Topics - without fold change
NET_DIR='./../network_analysis/'
TOPIC_DIR='./../analysate_by_2023data/topics/'
OUTPUT_DIR='./../data/patents/'

mkdir ${OUTPUT_DIR}4.topic.max_structured.time_split
python ${NET_DIR}gcn_data_preprocessing.py \
    -i ${OUTPUT_DIR}4.topic.max_structured/ \
    -t ${OUTPUT_DIR}3.graph/time_lines.txt \
    -m 8 \
    -o ${OUTPUT_DIR}4.topic.max_structured.time_split/train_valid \
    -o ${OUTPUT_DIR}4.topic.max_structured.time_split/test \
    -n 48 \

# Random clusters - without fold change
mkdir ${OUTPUT_DIR}5.random_cluster.wo_topic/clusters.max_structured.time_split
python ${NET_DIR}gcn_data_preprocessing.py \
    -i ${OUTPUT_DIR}5.random_cluster.wo_topic/clusters.max_structured/ \
    -t ${OUTPUT_DIR}3.graph/time_lines.txt \
    -m 8 \
    -o ${OUTPUT_DIR}5.random_cluster.wo_topic/clusters.max_structured.time_split/train_valid \
    -o ${OUTPUT_DIR}5.random_cluster.wo_topic/clusters.max_structured.time_split/test \
    -n 48 \
