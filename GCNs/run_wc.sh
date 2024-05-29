desc='desc'
seed=0
num_train_clusters=2000
num_valid_clusters=500
num_tset_clusters=500
epochs=100
lr=1e-2
device=3
batch_size=64
seq_len=12
pred_len=1

for media in patents papers news; do
  topic_dir='/data0/yejin/blockchain_data_2024/'$media'_co10/4.topic.max_structured.time_split/test'
  if [ "$media" = "patents" ]; then
    best_node_feature_type="betweenness closeness degree"
    best_lstm_hidden_size=4
    best_lstm_num_layers=3
    best_gru_hidden_size=4
    best_gru_num_layers=2
    best_agcrn_K=2
    best_agcrn_embedd_dim=16
    best_agcrn_out_channels=8
    best_a3tgcn2_out_channels=16
  elif [ "$media" = "papers" ]; then
    best_node_feature_type="betweenness closeness degree"
    best_lstm_hidden_size=32
    best_lstm_num_layers=1
    best_gru_hidden_size=16
    best_gru_num_layers=4
    best_agcrn_K=3
    best_agcrn_embedd_dim=4
    best_agcrn_out_channels=8
    best_a3tgcn2_out_channels=8
  elif [ "$media" = "news" ]; then
    best_node_feature_type="degree"
    best_lstm_hidden_size=4
    best_lstm_num_layers=1
    best_gru_hidden_size=4
    best_gru_num_layers=4
    best_agcrn_K=2
    best_agcrn_embedd_dim=4
    best_agcrn_out_channels=32
    best_a3tgcn2_out_channels=32
  else
    echo "Invalid media value. Please provide 'patents', 'papers', 'news'"
    exit 1
  fi

  for pred_len in 1 3 6 9 12; do # 1 3 6 9 12
    python3.7 -u main_wc_loader.py \
      --topic_dir $topic_dir \
      --seed $seed \
      --best_node_feature_type $best_node_feature_type \
      --best_lstm_hidden_size $best_lstm_hidden_size \
      --best_lstm_num_layers $best_lstm_num_layers \
      --best_gru_hidden_size $best_gru_hidden_size \
      --best_gru_num_layers $best_gru_num_layers \
      --best_agcrn_K $best_agcrn_K \
      --best_agcrn_embedd_dim $best_agcrn_embedd_dim \
      --best_agcrn_out_channels $best_agcrn_out_channels \
      --best_a3tgcn2_out_channels $best_a3tgcn2_out_channels \
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
