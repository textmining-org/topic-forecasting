seed=0

seq_len=12
desc='desc'

epochs=100
patience=5

results_path='./results/joi_2024'
metrics_file='metrics_trng_a3tgcn2.csv'
model=a3tgcn2

# 165.132.148.123
device=0      # 0 1 3
media=patents # patents papers news
cluster_dir='/data0/yejin/blockchain_data_2024/'$media'_co10/5.random_cluster/clusters.max_structured.time_split'

for num_train_clusters in 2000; do   # 100 500 1000 2000 4000 6000 8000
  for num_valid_clusters in 500; do  # 1000 2000
    for num_test_clusters in 500; do # 1000 2000
      for batch_size in 8 16 32; do
        for lr in 1e-2 1e-3 1e-4; do
          for node_feature_type in "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"; do
            for out_channels in 4 8 16 32; do # 4 8 16 32
              for pred_len in 1 3 6 9 12; do    # 1 3 6 9 12
                python3.7 -u main_trng_loader.py --seed $seed \
                  --cluster_dir $cluster_dir \
                  --model $model \
                  --node_feature_type $node_feature_type \
                  --epochs $epochs \
                  --patience $patience \
                  --device $device \
                  --num_train_clusters $num_train_clusters \
                  --num_valid_clusters $num_valid_clusters \
                  --num_test_clusters $num_test_clusters \
                  --lr $lr \
                  --media $media \
                  --batch_size $batch_size \
                  --seq_len $seq_len \
                  --pred_len $pred_len \
                  --early_stop \
                  --results_path $results_path \
                  --metrics_file $metrics_file \
                  --out_channels $out_channels \
                  --desc $desc >trng_$media'_'$model.log
              done
            done
          done
        done
      done
    done
  done
done
