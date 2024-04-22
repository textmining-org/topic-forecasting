media=papers

#cluster_dir='/Data2/yejin/blockchain_data_20230420/'$media'_co10/5.random_cluster/clusters.max_structured.time_split'
cluster_dir='/Data2/yejin/blockchain_data_2024/'$media'_co10/5.random_cluster/clusters.max_structured.time_split'

seed=0
#num_train_clusters=4000
#num_valid_clusters=1000
#num_tset_clusters=1000
num_train_clusters=100
num_valid_clusters=20
num_tset_clusters=20

epochs=500
lr=1e-3
device=0
batch_size=64
seq_len=12
pred_len=1

desc='desc'

# "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"
# "betweenness closeness" "betweenness degree"
# "closeness degree" "closeness"
# "betweenness" "degree"
# "betweenness closeness degree"

for node_feature_type in "degree"; do
  for model in a3tgcn2; do           # dcrnn tgcn agcrn a3tgcn a3tgcn2
    for lr in 1e-3; do               # 1e-2 1e-3 1e-4
      for pred_len in 1; do # 1 3 6 9 12
        python3 -u main_trng_loader.py --seed $seed \
          --cluster_dir $cluster_dir \
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
          --desc $desc >trng_$model.log
      done
    done
  done
done
