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

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    fcst_vis_save_path = os.path.join(results_path, 'fcst_vis')

    node_feature_type = '_'.join(args.node_feature_type)
    # (topic, time, node target) : (13, 64, 50)
    true_ys, fcst_ys = load_fcst_y(fcst_val_save_path, args.media, args.model, node_feature_type,
                                   args.num_train_clusters, args.desc)

    # (topic, time, target sum) : (13, 64, 1)
    true_ys = np.sum(true_ys, axis=2)
    fcst_ys = np.sum(fcst_ys, axis=2)

    print(true_ys.shape, fcst_ys.shape)

    for topic_num in range(true_ys.shape[0]):
        # plt.figure(figsize=(15, 7))
        # FIXME topic_num
        plt.plot(range(true_ys.shape[1]), true_ys[topic_num], label="ground truth", linestyle="--")
        plt.plot(range(true_ys.shape[1]), fcst_ys[topic_num], label=f'topic-{topic_num}')

        plt.title(f'{str(args.media).capitalize()} Topic-{topic_num} Forecasting based on {str(args.model).upper()}')
        plt.legend(loc='upper left')
        plt.ylabel("keywords frequency")
        plt.xlabel("timeline")

        filename = f'{args.media}_{args.model}_{node_feature_type}_{args.num_train_clusters}_{args.desc}_{topic_num}.png'
        # filename = f'papers_agcrn_betweenness_closeness_degree_1000_w_topic_w_fc_{topic_num}_inverted.png'
        plt.savefig(os.path.join(fcst_vis_save_path, filename))
        plt.clf()
