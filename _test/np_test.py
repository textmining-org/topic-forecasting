import numpy as np
import os

# fcst_val_save_path = '../GCNs/results/backup/20230414/fcst_val/'
#
# true_ys = np.load(os.path.join(fcst_val_save_path, 'papers_agcrn_closeness_degree_1000_w_topic_w_fc_fcst.npy'))
# fcst_ys = np.load(os.path.join(fcst_val_save_path, 'papers_agcrn_closeness_degree_1000_w_topic_w_fc_true.npy'))
#
# print(true_ys.shape)
# print(true_ys)

list1 = ['betweenness', 'closeness', 'degree']
list2 = ['betweenness', 'closeness', 'degree']

print(list1==list2)