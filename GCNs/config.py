import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='0')
parser.add_argument('--device_ids', type=int, default=[0, 1, 2, 3], action="append")
# parser.add_argument('--model', type=str, default='a3tgcn2')  # dcrnn tgcn agcrn a3tgcn a3tgcn2
parser.add_argument('--model', type=str, default='lstm')  # dcrnn tgcn agcrn a3tgcn a3tgcn2 lstm gru
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--node_feature_type', nargs='+', help='<Required> Set flag', default=['betweenness', 'closeness', 'degree'])
parser.add_argument('--topic_num', type=int, default=1)
parser.add_argument('--discard_index', type=int, default=0)
# parser.add_argument('--num_train_clusters', type=int, default=2000)
# parser.add_argument('--num_valid_clusters', type=int, default=1000)
# parser.add_argument('--num_test_clusters', type=int, default=1000)
parser.add_argument('--num_train_clusters', type=int, default=2000)
parser.add_argument('--num_valid_clusters', type=int, default=500)
parser.add_argument('--num_test_clusters', type=int, default=500)
parser.add_argument('--media', type=str, default='patents')  # patents papers news
# parser.add_argument('--cluster_dir', type=str, default='/Data2/yejin/blockchain_data_20230420/patents_co10/5.random_cluster/clusters.max_structured.time_split')
# parser.add_argument('--topic_dir', type=str, default='/Data2/yejin/blockchain_data_20230420/patents_co10/4.topic.max_structured.time_split/test')
# parser.add_argument('--cluster_dir', type=str, default='/Data2/yejin/blockchain_data_20230420/papers_co10/5.random_cluster/clusters.max_structured.time_split')
# parser.add_argument('--topic_dir', type=str, default='/Data2/yejin/blockchain_data_20230420/papers_co10/4.topic.max_structured.time_split/test')
parser.add_argument('--cluster_dir', type=str, default='/data0/yejin/blockchain_data_2024/patents_co10/5.random_cluster/clusters.max_structured.time_split')
parser.add_argument('--topic_dir', type=str, default='/data0/yejin/blockchain_data_2024/patents_co10/4.topic.max_structured.time_split/test')

# model
parser.add_argument('--out_channels', type=int, default=8)
parser.add_argument('--out_size', type=int, default=1)
# filter size : DCRNN, AGCRN
parser.add_argument('--K', type=int, default=2)
# node embedding dimension AGCRN
parser.add_argument('--embedd_dim', type=int, default=16)
# A3TGCN
parser.add_argument('--periods', type=int, default=1)
# A3TGCN2
parser.add_argument('--seq_len', type=int, default=12)
parser.add_argument('--pred_len', type=int, default=3)
# LSTM, GRU
parser.add_argument('--hidden_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)

# Early Stopping
parser.add_argument('--early_stop', action='store_true', help='', default=False)
parser.add_argument('--patience', type=int, default=10)

# etc
parser.add_argument('--results_path', type=str, default='./results/joi_2024')
parser.add_argument('--metrics_file', type=str, default='metrics_trng_a3tgcn2.csv')
parser.add_argument('--desc', type=str, default='desc')

# best settings
parser.add_argument('--best_node_feature_type', nargs='+', help='<Required> Set flag', default=['betweenness', 'closeness', 'degree'])
parser.add_argument('--best_lstm_hidden_size', type=int, default=4)
parser.add_argument('--best_lstm_num_layers', type=int, default=3)
parser.add_argument('--best_gru_hidden_size', type=int, default=4)
parser.add_argument('--best_gru_num_layers', type=int, default=2)
parser.add_argument('--best_agcrn_K', type=int, default=2)
parser.add_argument('--best_agcrn_embedd_dim', type=int, default=16)
parser.add_argument('--best_agcrn_out_channels', type=int, default=8)
parser.add_argument('--best_a3tgcn2_out_channels', type=int, default=16)

def get_config():
    return parser.parse_args()


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
