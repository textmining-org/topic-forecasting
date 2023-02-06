
seed=0
# "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"
for node_feature_type in "betweenness" "degree" "closeness"; do #  betweenness closeness degree
  for model in a3tgcn; do                # 'dcrnn' 'tgcn' 'agcrn' 'a3tgcn'
    for epochs in 200; do
      for device in 3; do
        for lr in 1e-3; do
          python3 -u main.py --seed $seed --model $model --node_feature_type $node_feature_type --epochs $epochs --device $device --lr $lr
        done
      done
    done
  done
done
