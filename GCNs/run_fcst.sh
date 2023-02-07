seed=0

for node_feature_type in "betweenness closeness" "betweenness degree" "closeness degree" "betweenness closeness degree" "betweenness" "degree" "closeness"; do
  for model in dcrnn tgcn agcrn a3tgcn; do # dcrnn tgcn agcrn a3tgcn
    for device in 0; do
      python3 -u main_fcst.py --seed $seed --model $model --node_feature_type $node_feature_type --device $device
    done
  done
done
