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
import pandas as pd

if __name__ == "__main__":
    # set configuration
    args = get_config()
    print(args)

    results_path = os.path.abspath(args.results_path)
    fcst_val_save_path = os.path.join(results_path, 'fcst_val')
    fcst_wc_save_path = os.path.join(results_path, 'fcst_wc')

    node_feature_type = '_'.join(args.best_node_feature_type)
    # true_ys, fcst_ys_lstm, fcst_ys_gru, fcst_ys_agcrn, fcst_ys_a3tgcn2 : (topic number, batch, node index, forecasting results) = (19, 22, 20, 1)
    # xs : (topic number, time step, node index, look-back window) = (19, 22, 20, 12)
    true_ys, fcst_ys_lstm, xs = load_fcst_y(fcst_val_save_path, args, 'lstm', load_x=True)
    _, fcst_ys_gru, _ = load_fcst_y(fcst_val_save_path, args, 'gru', load_x=True)
    _, fcst_ys_agcrn, _ = load_fcst_y(fcst_val_save_path, args, 'agcrn', load_x=True)
    _, fcst_ys_a3tgcn2, _ = load_fcst_y(fcst_val_save_path, args, 'a3tgcn2', load_x=True)

    print(true_ys.shape, fcst_ys_lstm.shape, fcst_ys_gru.shape, fcst_ys_agcrn.shape, fcst_ys_a3tgcn2.shape, xs.shape)
    true_ys_total = np.concatenate((xs, true_ys), axis=3)

    print(true_ys.shape, fcst_ys_lstm.shape, fcst_ys_gru.shape, fcst_ys_agcrn.shape, fcst_ys_a3tgcn2.shape, xs.shape, true_ys_total.shape)

    for topic_num in range(true_ys_total.shape[0]):
        node_indices_path = os.path.join(args.topic_dir, str(topic_num+1))
        node_indices_path = os.path.join(node_indices_path, 'node_indices.tsv')
        node_indices_df = pd.read_csv(node_indices_path, sep='\t', header=None, names=["node_index", "node_name"])

        # Ensure node indices are aligned with the data
        # node_indices_df = node_indices_df.sort_values("index").reset_index(drop=True)
        print(node_indices_df)

        # Create a DataFrame to store the results
        results_df = pd.DataFrame()

        # Add node names to the results DataFrame
        results_df['node_index'] = node_indices_df['node_index']
        results_df['node_name'] = node_indices_df['node_name']

        for batch_idx in range(true_ys_total.shape[1]):
            topic_true_ys = true_ys_total[topic_num][batch_idx]
            topic_fcst_ys_lstm = fcst_ys_lstm[topic_num][batch_idx]
            topic_fcst_ys_gru = fcst_ys_gru[topic_num][batch_idx]
            topic_fcst_ys_agcrn = fcst_ys_agcrn[topic_num][batch_idx]
            topic_fcst_ys_a3tgcn2 = fcst_ys_a3tgcn2[topic_num][batch_idx]
            print(topic_num, batch_idx, topic_true_ys.shape, topic_fcst_ys_lstm.shape, topic_fcst_ys_gru.shape,
                  topic_fcst_ys_agcrn.shape, topic_fcst_ys_a3tgcn2.shape)

            # Add true values to the DataFrame
            for i in range(topic_true_ys.shape[1]):
                results_df[f'true_ys_{i}'] = topic_true_ys[:, i]

            # Add forecast values for each model to the DataFrame
            for i in range(topic_fcst_ys_lstm.shape[1]):
                results_df[f'fcst_ys_lstm_{i}'] = topic_fcst_ys_lstm[:, i]
                results_df[f'fcst_ys_gru_{i}'] = topic_fcst_ys_gru[:, i]
                results_df[f'fcst_ys_agcrn_{i}'] = topic_fcst_ys_agcrn[:, i]
                results_df[f'fcst_ys_a3tgcn2_{i}'] = topic_fcst_ys_a3tgcn2[:, i]

            # Save the DataFrame to a TSV file
            output_file_path = os.path.join(fcst_wc_save_path, f'{args.media}_topic_{topic_num}_pred_len_{args.pred_len}_results.tsv')
            results_df.to_csv(output_file_path, sep='\t', index=False)