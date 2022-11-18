discard_index=0

for seed in 0; do
  for media in 'patents'; do
    for topic_num in 1 2 3 4 5 6 7 8 9 10 11; do  # patents(1-11), papers(1-8)
      for model in 'dcrnn' 'tgcn' 'agcrn' 'a3tgcn'; do # dcrnn tgcn agcrn
        for epochs in 200; do
          for device in 1; do
            for lr in 1e-3; do
              python3 -u main.py --seed $seed --media $media --topic_num $topic_num --model $model --epochs $epochs --device $device --lr $lr --discard_index $discard_index
            done
          done
        done
      done
    done
  done
done
