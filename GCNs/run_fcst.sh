media=papers

#topic_dir='/Data2/yejin/blockchain_data_20230420/'$media'_co10/4.topic.max_structured.time_split/test'
topic_dir='/Data2/yejin/blockchain_data_2024/'$media'_co10/4.topic.max_structured.time_split/test'
desc='desc'

seed=0
num_train_clusters=1000
num_valid_clusters=2000
num_tset_clusters=2000

epochs=100

lr=1e-3

device=3
batch_size=32
seq_len=12
pred_len=1

# "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"
# "betweenness closeness" "betweenness degree"
# "closeness degree" "betweenness closeness degree"
# "betweenness" "degree"
# "closeness"

for node_feature_type in "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"; do
  for model in a3tgcn2; do   # dcrnn tgcn agcrn a3tgcn a3tgcn2
    for lr in 1e-2 1e-3 1e-4; do       # 1e-2 1e-3 1e-4
      for pred_len in 1 3 6 9 12; do # 1 3 6 9 12
        python3 -u main_fcst_loader.py --seed $seed \
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
          --desc $desc
      done
    done
  done
done
