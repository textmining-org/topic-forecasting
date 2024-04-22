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

    node_feature_type = '_'.join(args.node_feature_type)
    # true_ys, fcst_ys : (topic, batch, node, pred_len) = (11, 17, 20, 6)
    # xs : (topic, time, node, seq_len) = (11, 17, 20, 12)
    true_ys, fcst_ys, xs = load_fcst_y(fcst_val_save_path, args, load_x=True)
    print(true_ys.shape, fcst_ys.shape, xs.shape)

    # true_ys, fcst_ys : (topic, batch, pred_len) = (11, 17, 6)
    # xs : (topic, batch, seq_len) = (11, 17, 12)
    # true_ys_total(xs+true_ys) : (11, 17, 18)
    true_ys = np.sum(true_ys, axis=2)
    fcst_ys = np.sum(fcst_ys, axis=2)
    xs = np.sum(xs, axis=2)
    true_ys_total = np.concatenate((xs, true_ys), axis=2)

    print(true_ys.shape, fcst_ys.shape, xs.shape, true_ys_total.shape)

    for topic_num in range(true_ys_total.shape[0]):
        for batch_idx in range(true_ys_total.shape[1]):
            topic_true_ys = true_ys_total[topic_num][batch_idx]
            topic_fcst_ys = fcst_ys[topic_num][batch_idx]
            print(topic_num, batch_idx, topic_true_ys.shape, topic_fcst_ys.shape)

            # FIXME topic_num
            plt.plot(range(len(topic_true_ys)), topic_true_ys, label="ground truth", linestyle="--")
            # plt.plot(range(len(topic_true_ys)-len(topic_fcst_ys), len(topic_true_ys)), topic_fcst_ys, label=f'topic-{topic_num}|batch idx-{batch_idx}')


            if topic_fcst_ys.shape[0] == 1:
                temp_range = range(len(topic_true_ys) - len(topic_fcst_ys), len(topic_true_ys))
                plt.plot(range(len(topic_true_ys) - len(topic_fcst_ys), len(topic_true_ys)), topic_fcst_ys,
                         label='forecasting', marker="o")
            else:
                plt.plot(range(len(topic_true_ys) - len(topic_fcst_ys), len(topic_true_ys)), topic_fcst_ys,
                         label='forecasting')

            # plt.title(f'{str(args.media).capitalize()} Topic-{topic_num} Forecasting based on {str(args.model).upper()}')
            plt.title(f'Topic-{topic_num + 1} Trend Forecasting')
            plt.legend(loc='upper left')
            plt.ylabel("Word count")
            plt.xlabel("Time")

            filename = f'{args.media}_{args.model}_{node_feature_type}_{args.epochs}_{args.batch_size}_{args.lr}_{args.num_train_clusters}_{args.num_valid_clusters}_{args.num_test_clusters}_{args.seq_len}_{args.pred_len}_{args.desc}_{str(topic_num + 1).zfill(2)}_{str(batch_idx).zfill(2)}'
            # filename = f'papers_agcrn_betweenness_closeness_degree_1000_w_topic_w_fc_{topic_num}_inverted.png'
            plt.savefig(os.path.join(fcst_vis_save_path, filename + '.svg'))
            plt.savefig(os.path.join(fcst_vis_save_path, filename + '.png'))
            plt.clf()
