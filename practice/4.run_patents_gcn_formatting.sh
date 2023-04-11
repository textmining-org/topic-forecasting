# Topics - without fold change
mkdir ./patents_co10/4.topic.max_structured.time_split
python ./../network_analysis/gcn_data_preprocessing.py \
    -i ./patents_co10/4.topic.max_structured/ \
    -t ./patents_co10/3.graph/time_line.txt \
    -m 8 \
    -o ./patents_co10/4.topic.max_structured.time_split/train \
    -o ./patents_co10/4.topic.max_structured.time_split/valid \
    -o ./patents_co10/4.topic.max_structured.time_split/test \
    -n 48 \
    -n 60 \

# Topics - with fold change
mkdir ./patents_co10/4.topic.max_structured.fc_converted.time_split
python ./../network_analysis/gcn_data_preprocessing.py \
    -i ./patents_co10/4.topic.max_structured/ \
    -t ./patents_co10/3.graph/fc_time_line.txt \
    -f \
    -m 8 \
    -o ./patents_co10/4.topic.max_structured.fc_converted.time_split/train \
    -o ./patents_co10/4.topic.max_structured.fc_converted.time_split/valid \
    -o ./patents_co10/4.topic.max_structured.fc_converted.time_split/test \
    -n 48 \
    -n 60 \

# Random clusters - without fold change
mkdir ./patents_co10/5.random_cluster/clusters.max_structured.time_split
python ./../network_analysis/gcn_data_preprocessing.py \
    -i ./patents_co10/5.random_cluster/clusters.max_structured/ \
    -t ./patents_co10/3.graph/time_line.txt \
    -m 8 \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.time_split/train \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.time_split/valid \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.time_split/test \
    -n 48 \
    -n 60 \

# Random clusters - with fold change
mkdir ./patents_co10/5.random_cluster/clusters.max_structured.fc_converted.time_split
python ./../network_analysis/gcn_data_preprocessing.py \
    -i ./patents_co10/5.random_cluster/clusters.max_structured/ \
    -t ./patents_co10/3.graph/fc_time_line.txt \
    -f \
    -m 8 \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.fc_converted.time_split/train \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.fc_converted.time_split/valid \
    -o ./patents_co10/5.random_cluster/clusters.max_structured.fc_converted.time_split/test \
    -n 48 \
    -n 60 \
