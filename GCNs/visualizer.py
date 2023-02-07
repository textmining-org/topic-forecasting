import matplotlib.pyplot as plt
import numpy as np
import argparse
from config import get_config
from _commons.utils import load_fcst_y
import os

if __name__ == "__main__":
    # set configuration
    args = get_config()

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    fcst_vis_save_path = os.path.join(results_path, 'fcst_vis')

    node_feature_type = '_'.join(args.node_feature_type)
    # (topic, time, node target) : (13, 64, 50)
    true_ys, fcst_ys = load_fcst_y(fcst_val_save_path, args.media, args.model, node_feature_type)

    # (topic, time, target sum) : (13, 64, 1)
    true_ys = np.sum(true_ys, axis=2)
    fcst_ys = np.sum(fcst_ys, axis=2)

    for topic_num in range(true_ys.shape[0]):
        # plt.figure(figsize=(15, 7))
        plt.plot(range(true_ys.shape[1]), true_ys[topic_num], label="ground truth", linestyle="--")
        plt.plot(range(true_ys.shape[1]), fcst_ys[topic_num], label=f'topic-{topic_num}')

        plt.title(f'{str(args.media).capitalize()} Topic-{topic_num} Forecasting based on {str(args.model).upper()}')
        plt.legend(loc='upper left')
        plt.ylabel("keywords frequency")
        plt.xlabel("timeline")
        # plt.tight_layout()

        filename = f'{args.media}_{args.model}_{node_feature_type}_{topic_num}.png'
        plt.savefig(os.path.join(fcst_vis_save_path, filename))
        plt.clf()

