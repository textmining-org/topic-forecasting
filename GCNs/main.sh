discard_index=12

for seed in 0; do
  for media in 'papers'; do
    for topic_num in 1 2 3 4 5 6 7 8; do  # patents(1-11), papers(1-8)
      for model in 'dcrnn'; do # dcrnn tgcn agcrn
        for epochs in 200; do
          for device in 0; do
            for lr in 1e-3; do
              python3 -u main.py --seed $seed --media $media --topic_num $topic_num --model $model --epochs $epochs --device $device --lr $lr --discard_index $discard_index
            done
          done
        done
      done
    done
  done
done
