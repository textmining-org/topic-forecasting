#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd


def _revert_fold_change_(fc_arr,base_arr,base=np.e,pseudocount=1.0):
    return ((base**fc_arr)*(base_arr+pseudocount))-pseudocount


# fc_arr (TOPIC,TIME,WORD) : FC data topic wordcount data (time 1_0 to t_t-1)
# base_arr (TOPIC,TIME,WORD) : raw topic wordcount data (time 0 to t-1)
def revert_fc(fc_arr,base_arr,output_file=None,base=np.e,pseudocount=1.0):
    assert base_arr.shape[0] == fc_arr.shape[0]
    assert base_arr.shape[1] == fc_arr.shape[1]
    assert base_arr.shape[2] == fc_arr.shape[2]
    
    reverted = _revert_fold_change_(
        fc_arr=fc_arr,
        base_arr=base_arr,
        base=base,
        pseudocount=pseudocount)
    if output_file:
        np.save(output_file,reverted)
    return reverted



def parse_match_topic(fc_true_file,fc_pred_file,topic_data_dir,topic_data_fc_dir):
    fc_true = np.load(fc_true_file)
    fc_pred = np.load(fc_pred_file)
    base_arr_dict = {i:np.load(os.path.join(topic_data_dir,i,'word_count.node_targets.npy')) for i in os.listdir(topic_data_dir)}
    base_fc_arr_dict = {i:np.load(os.path.join(topic_data_fc_dir,i,'word_count.node_targets.npy')) for i in os.listdir(topic_data_fc_dir)}
    topic_ord_ser = pd.Series([None]*fc_true.shape[0],index=range(fc_true.shape[0]))
    for idx in range(fc_true.shape[0]):
        for topic_id, _arr in base_fc_arr_dict.items():
            if np.allclose(_arr[-1,:],fc_true[idx,-1,:],rtol=1e-04, atol=1e-04):
                topic_ord_ser[idx]=topic_id
                break
#     return fc_true, fc_pred, base_arr_dict, base_fc_arr_dict, topic_ord_ser
    base_arr = np.stack([base_arr_dict[topic_id] for topic_id in list(topic_ord_ser.values)])
    return fc_pred, base_arr, topic_ord_ser
    
    

def main(fc_true_file,fc_pred_file,topic_data_dir,topic_data_fc_dir,output=None,base_arr_margins=(1,0)):
    
    fc_pred, base_arr, topic_ord_ser = parse_match_topic(
        fc_true_file,fc_pred_file,topic_data_dir,topic_data_fc_dir)
    _time_start = base_arr_margins[0]
    _time_end = base_arr.shape[1]-base_arr_margins[1]
    _base_arr_margined_ = base_arr[:,:,:]
    fc_pred_rev = revert_fc(
        fc_arr=fc_pred,
        base_arr=base_arr[:,_time_start:_time_end,:],
        output_file=None,
        base=np.e,pseudocount=1.0)
    if output:
        np.save(output,fc_pred_rev)
    # tmp
    return fc_pred_rev, fc_pred, base_arr, base_arr[:,_time_start:_time_end,:], topic_ord_ser


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reversion of FC data')
    parser.add_argument('-ft','--fc_true',help='FC true array')
    parser.add_argument('-fp','--fc_pred',help='FC pred array')
    parser.add_argument('-td','--topic_dir',help='Raw topic directory without FC')
    parser.add_argument('-tf','--topic_fc_dir',help='Raw topic directory with FC')
    parser.add_argument('-o','--output_file',help='Output file')
    parser.add_argument('-t0','--start_point_margin',type=int,default=1,help='Index of starting time point to use for topic data')
    parser.add_argument('-t1','--end_point_margin',type=int,default=0,help=' of end time point to use for topic data')
    args=parser.parse_args()
    
    main(
        fc_true_file=args.fc_true,
        fc_pred_file=args.fc_pred,
        topic_data_dir=args.topic_dir,
        topic_data_fc_dir=args.topic_fc_dir,
        output=args.output,
        base_arr_margins=(t0,t1),
    )
    
    