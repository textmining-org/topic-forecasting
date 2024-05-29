# MSE 기준 최적 조합 node feature
# lstm
# * patents
# node_feature_type="betweenness closeness degree"
# hidden_size=4
# num_layers=3
# * papers
# node_feature_type="betweenness closeness degree"
# hidden_size=32
# num_layers=1
# * news
# node_feature_type="degree"
# hidden_size=4
# num_layers=1

# gru
# * patents
# node_feature_type="betweenness closeness degree"
# hidden_size=4
# num_layers=2
# * papers
# node_feature_type="betweenness closeness degree"
# hidden_size=16
# num_layers=4
# * news
# node_feature_type="degree"
# hidden_size=4
# num_layers=4

seed=0
num_train_clusters=2000
num_valid_clusters=500
num_tset_clusters=500
epochs=100
lr=1e-2
batch_size=64
seq_len=12
results_path='./results/joi_2024'
metrics_file='metrics_fcst_rnns.csv'

device=0
model=gru
media=news # patents papers news
topic_dir='/data0/yejin/blockchain_data_2024/'$media'_co10/4.topic.max_structured.time_split/test'
node_feature_type="degree"
hidden_size=4
num_layers=4

for pred_len in 1 3 6 9 12; do # 1 3 6 9 12
  python3.7 -u main_fcst_loader.py --seed $seed \
    --topic_dir $topic_dir \
    --model $model \
    --node_feature_type $node_feature_type \
    --epochs $epochs \
    --device $device \
    --num_train_clusters $num_train_clusters \
    --num_valid_clusters $num_valid_clusters \
    --num_test_clusters $num_tset_clusters \
    --lr $lr \
    --media $media \
    --batch_size $batch_size \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --hidden_size $hidden_size \
    --num_layers $num_layers \
    --results_path $results_path \
    --metrics_file $metrics_file
done
