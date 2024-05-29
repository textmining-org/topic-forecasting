import os
import sys

PLF_DIR = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
sys.path.append(PLF_DIR)
DEPS = [os.path.join(PLF_DIR, i) for i in os.listdir(PLF_DIR)]
sys.path.extend(DEPS)

import matplotlib.pyplot as plt
import numpy as np
from config import get_config
from _commons.utils import load_fcst_y

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    fcst_vis_save_path = os.path.join(results_path, 'fcst_vis')

    node_feature_type = '_'.join(args.best_node_feature_type)
    # true_ys, fcst_ys_lstm, fcst_ys_gru, fcst_ys_agcrn, fcst_ys_a3tgcn2 : (topic number, batch, node index, forecasting results) = (19, 22, 20, 1)
    # xs : (topic number, time step, node index, look-back window) = (19, 22, 20, 12)
    true_ys, fcst_ys_lstm, xs = load_fcst_y(fcst_val_save_path, args, 'lstm', load_x=True)
    _, fcst_ys_gru, _ = load_fcst_y(fcst_val_save_path, args, 'gru', load_x=True)
    _, fcst_ys_agcrn, _ = load_fcst_y(fcst_val_save_path, args, 'agcrn', load_x=True)
    _, fcst_ys_a3tgcn2, _ = load_fcst_y(fcst_val_save_path, args, 'a3tgcn2', load_x=True)

    print(true_ys.shape, fcst_ys_lstm.shape, fcst_ys_gru.shape, fcst_ys_agcrn.shape, fcst_ys_a3tgcn2.shape, xs.shape)

    # 모든 node(토픽의 키워드)의 값을 합계
    true_ys = np.sum(true_ys, axis=2)
    fcst_ys_lstm = np.sum(fcst_ys_lstm, axis=2)
    fcst_ys_gru = np.sum(fcst_ys_gru, axis=2)
    fcst_ys_agcrn = np.sum(fcst_ys_agcrn, axis=2)
    fcst_ys_a3tgcn2 = np.sum(fcst_ys_a3tgcn2, axis=2)
    xs = np.sum(xs, axis=2)
    true_ys_total = np.concatenate((xs, true_ys), axis=2)
    # true_ys, fcst_ys : (topic number, batch, forecasting results) = (19, 22, 1)
    # xs : (topic number, batch, look-back window) = (19, 22, 12)
    # true_ys_total(xs+true_ys) : (19, 22, 13)

    print(true_ys.shape, fcst_ys_lstm.shape, fcst_ys_gru.shape, fcst_ys_agcrn.shape, fcst_ys_a3tgcn2.shape, xs.shape, true_ys_total.shape)

    colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99']
    markers = ['o', 's', '^', 'd']

    for topic_num in range(true_ys_total.shape[0]):
        for batch_idx in range(true_ys_total.shape[1]):
            topic_true_ys = true_ys_total[topic_num][batch_idx]
            topic_fcst_ys_lstm = fcst_ys_lstm[topic_num][batch_idx]
            topic_fcst_ys_gru = fcst_ys_gru[topic_num][batch_idx]
            topic_fcst_ys_agcrn = fcst_ys_agcrn[topic_num][batch_idx]
            topic_fcst_ys_a3tgcn2 = fcst_ys_a3tgcn2[topic_num][batch_idx]
            print(topic_num, batch_idx, topic_true_ys.shape, topic_fcst_ys_lstm.shape, topic_fcst_ys_gru.shape,
                  topic_fcst_ys_agcrn.shape, topic_fcst_ys_a3tgcn2.shape)

            plt.figure(figsize=(10, 6))  # 그래프 크기 조정
            plt.plot(range(len(topic_true_ys)), topic_true_ys, label="Ground Truth", linestyle="--", linewidth=2,
                     color='k')

            models = ['LSTM', 'GRU', 'AGCRN', 'A3TGCN']
            fcst_data = [topic_fcst_ys_lstm, topic_fcst_ys_gru, topic_fcst_ys_agcrn, topic_fcst_ys_a3tgcn2]

            for i, (model, fcst_ys) in enumerate(zip(models, fcst_data)):
                if fcst_ys.shape[0] == 1:
                    plt.plot(range(len(topic_true_ys) - len(fcst_ys), len(topic_true_ys)), fcst_ys,
                             label=model, color=colors[i], marker=markers[i], linewidth=2, markersize=8)
                else:
                    plt.plot(range(len(topic_true_ys) - len(fcst_ys), len(topic_true_ys)), fcst_ys,
                             label=model, color=colors[i], marker=markers[i], linewidth=2)

            plt.title(f'Topic-{topic_num + 1} Trend Forecasting', fontsize=16)
            plt.legend(loc='upper left', fontsize=12)
            plt.ylabel("Word Count", fontsize=14)
            plt.xlabel("Time", fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True)

            filename = f'{args.media}_all_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{str(topic_num + 1).zfill(2)}_{str(batch_idx).zfill(2)}'
            plt.tight_layout()
            plt.savefig(os.path.join(fcst_vis_save_path, filename + '.svg'), dpi=1000)
            plt.savefig(os.path.join(fcst_vis_save_path, filename + '.png'), dpi=1000)
            plt.clf()
