"""
For data preprocessing of GCN data
1. (optional) Fold change calculation
2. Split dataset by timepoint
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import json
import tqdm
import _multi
import argparse
import re


# returns log_base(arr1+pseudocount/arr2+pseudocount)
def _fold_change_(arr1,arr2,base=np.e,pseudocount=1.0):
    return np.log((arr1+pseudocount)/(arr2+pseudocount))/np.log(base)


# fc_arr = log_base(arr1+pseudocount/arr2+pseudocount)
# base_arr = the original arr2
# returns arr1
def _revert_fold_change_(fc_arr,base_arr,base=np.e,pseudocount=1.0):
    return ((base**fc_arr)*(base_arr+pseudocount))-pseudocount


# sort files
def sort_gcn_feature_dir_file(data_dir):
    _input_dir = os.path.abspath(data_dir)
    other_fs = [os.path.join(_input_dir,i) for i in os.listdir(_input_dir) if not i.startswith('.') and not i.startswith('_')]
    # sort by node info, edge info and others    
    
    get_file_prfx = lambda x,_suf: os.path.split(x)[-1][:-len(_suf)] if x.endswith(_suf) else False
    node_fs = {}# {PREFIX:(FILE1,FILE2)}
    for _idx, _node_sfx in enumerate([".node_targets.npy",".node_attributes.txt"]):
        for _f in glob.glob(_input_dir+'/*'+_node_sfx):
            other_fs.remove(_f)
            prfx = get_file_prfx(_f,_node_sfx)
            if prfx:
                if prfx not in node_fs:
                    node_fs[prfx] = [False,False]
                node_fs[prfx][_idx] = _f
    edge_fs = {}# {PREFIX:(FILE1,FILE2,FILE3)}
    for _idx, _edge_sfx in enumerate([".edge_attributes.txt",".edge_indices.json",".edge_weights.json"]):
        for _f in glob.glob(_input_dir+'/*'+_edge_sfx):
            other_fs.remove(_f)
            prfx = get_file_prfx(_f,_edge_sfx)
            if prfx:
                if prfx not in edge_fs:
                    edge_fs[prfx] = [False,False,False]
                edge_fs[prfx][_idx] = _f
    
    return node_fs, edge_fs, other_fs


def __read_node_fs__(prfx):
    # fold change
    assert (os.path.isfile(prfx+'.node_targets.npy') and os.path.isfile(prfx+'.node_attributes.txt'))
    val_arr = np.load(prfx+'.node_targets.npy')
    with open(prfx+'.node_attributes.txt','rb') as f:
        attrb_l = f.read().decode().split() # [TIME_KEY]
    while '' in attrb_l:
        attrb_l.remove('')
    return val_arr, attrb_l
    
    
def __write_node_fs__(val_arr,attrb_l,prfx):
    np.save(prfx+'.node_targets.npy',val_arr)
    with open(prfx+'.node_attributes.txt','wb') as f:
        f.write('\n'.join(attrb_l).encode())

        
def __read_edge_fs__(prfx):
    assert (os.path.isfile(prfx+'.edge_attributes.txt') and os.path.isfile(prfx+'.edge_indices.json') and os.path.isfile(prfx+'.edge_weights.json'))
    with open(prfx+'.edge_attributes.txt','rb') as f:
        attrb_l = f.read().decode().split() # [TIME_ATTRIUBTE_KEY]
    while '' in attrb_l:
        attrb_l.remove('')
    with open(prfx+'.edge_indices.json','rb') as f:
        _idx_dict = json.loads(f.read().decode()) # {TIME_KEY:[[source_idx],[target_idx]]}
    with open(prfx+'.edge_weights.json','rb') as f:
        _weight_dict = json.loads(f.read().decode()) # {TIME_KEY:[weight_val]}
    return attrb_l,_idx_dict,_weight_dict


def __write_edge_fs__(attrb_l,_idx_dict,_weight_dict,prfx):
    with open(prfx+'.edge_attributes.txt','wb') as f:
        f.write('\n'.join(attrb_l).encode())
    with open(prfx+'.edge_indices.json','wb') as f:
        f.write(json.dumps(_idx_dict).encode())
    with open(prfx+'.edge_weights.json','wb') as f:
        f.write(json.dumps(_weight_dict).encode())


# log(X_t/X_t-1)
def convert_fold_change(cluster_data_dir,output_dir,base=np.e,pseudocount=1.0,wo_node_fc=False,wo_edge_fc=False):
    _o_dir = os.path.abspath(output_dir)
    os.makedirs(_o_dir,exist_ok=True)
    
    node_fs,edge_fs,other_fs = sort_gcn_feature_dir_file(cluster_data_dir)
    # Node val
    for _prfx, _fs in node_fs.items():
        
            # fold change
        val_arr,attrb_l = __read_node_fs__(os.path.join(cluster_data_dir,_prfx))
        assert val_arr.shape[0] == len(attrb_l) # WARNING check numpy shape
        fc_val_arr = []
        fc_attrb_l = attrb_l[1:]
        for idxt1 in range(1,len(attrb_l)):
            idxt0 = idxt1-1
            # WARNING check numpy shape
            val0 = val_arr[idxt0,:]
            val1 = val_arr[idxt1,:]
            if not wo_node_fc:
                fc_val = _fold_change_(val1,val0,base=base,pseudocount=pseudocount).reshape(1,-1)
            else:
                fc_val = val1.reshape(1,-1)
            fc_val_arr.append(fc_val)
        del attrb_l
        del val_arr
        
        fc_val_arr = np.concatenate(fc_val_arr,axis=0)
        _out_prfx=os.path.join(_o_dir,_prfx)
        __write_node_fs__(val_arr=fc_val_arr,attrb_l=fc_attrb_l,prfx=_out_prfx)
            
    # Edge val
    for _prfx, _fs in edge_fs.items():
        attrb_l,_idx_dict,_weight_dict = __read_edge_fs__(os.path.join(cluster_data_dir,_prfx))

        assert set(attrb_l) == set(_idx_dict.keys())
        assert set(attrb_l) == set(_weight_dict.keys())
            
        # FC conversion
        fc_wgt_dict = {}
        fc_idx_dict = {}
        for _idx in range(1,len(attrb_l)):
            _t_1 = attrb_l[_idx]
            _t_0 = attrb_l[_idx-1]
            if not wo_edge_fc:
                # WARNING, check index
                _t_0_ser = pd.Series(_weight_dict[_t_0],
                                     index=list(zip(_idx_dict[_t_0][0],_idx_dict[_t_0][1]))) #((SRC1_no,SRC2_no))
                _t_1_ser = pd.Series(_weight_dict[_t_1],
                                     index=list(zip(_idx_dict[_t_1][0],_idx_dict[_t_1][1])))
                _whole_idx = sorted(list(set(_t_0_ser.index).union(set(_t_1_ser.index))))
                # WARNING, check index
                _t_0_ser = pd.Series(_t_0_ser,index=_whole_idx)
                _t_0_ser = _t_0_ser.fillna(0.0)
                _t_1_ser = pd.Series(_t_1_ser,index=_whole_idx)
                _t_1_ser = _t_1_ser.fillna(0.0)
                fc_arr = _fold_change_(
                    np.array(_t_1_ser.values),
                    np.array(_t_0_ser.values),
                    base=base,
                    pseudocount=pseudocount
                )
                fc_wgt_dict[_t_1] = fc_arr.reshape(-1).tolist()
                fc_idx_dict[_t_1] = [[i[0] for i in _whole_idx],[i[1] for i in _whole_idx]]
            else:
                fc_wgt_dict[_t_1] = _weight_dict[_t_1]
                fc_idx_dict[_t_1] = _idx_dict[_t_1]
            
        fc_attrb_l=attrb_l[1:]
        # write
        _out_prfx=os.path.join(_o_dir,_prfx)
        __write_edge_fs__(attrb_l=fc_attrb_l,_idx_dict=fc_idx_dict,_weight_dict=fc_wgt_dict,prfx=_out_prfx)
        del attrb_l
        del _idx_dict
        del _weight_dict
    # Other files
    for _f in other_fs:
        os.system('cp %s %s'%(
            _f,
            os.path.join(_o_dir,os.path.split(_f)[-1]),
        ))

def convert_fold_change_batch(master_cluster_data_dir,
                              master_output_dir,
                              base=np.e,
                              pseudocount=1.0,
                              wo_node_fc=False,
                              wo_edge_fc=False,
                              multiprocess:int=8,
                             ):
    m_i_d = os.path.abspath(master_cluster_data_dir)
    m_o_d = os.path.abspath(master_output_dir)
    os.makedirs(m_o_d,exist_ok=True)
    
#     for sub_dir in tqdm.tqdm(os.listdir(m_i_d)):
#         convert_fold_change(cluster_data_dir=os.path.join(m_i_d,sub_dir),
#                             output_dir=os.path.join(m_o_d,sub_dir),
#                             base=base,pseudocount=pseudocount)
    fn_arg_list = []
    fn_kwarg_list = []
    for sub_dir in os.listdir(m_i_d):
        fn_arg_list.append(tuple())
        fn_kwarg_list.append(dict(
            cluster_data_dir=os.path.join(m_i_d,sub_dir),
            output_dir=os.path.join(m_o_d,sub_dir),
            base=base,pseudocount=pseudocount,
            wo_node_fc=wo_node_fc,wo_edge_fc=wo_edge_fc,
        ))
        
    fn = convert_fold_change
    fn_args = _multi.argument_generator(fn_arg_list)
    fn_kwargs = _multi.keyword_argument_generator(fn_kwarg_list)
    _multi.multi_function_execution(
        fn=fn,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        max_processes=multiprocess,
        collect_result=False,
    )
        
        
# X_0 ~ X_n --> (X_0~X_a),(X_a~X_b),(X_b~X_n)
def split_feature_time(cluster_data_dir,
                       output_dirs:list=['./train_timeline','./val_timeline','./test_timeline'],
                       whole_timeline_file:str='./time_line.txt',
                       dir_portion_n_list:list=[48,60],
                       dir_portion_ratio_list:list=[0.7,0.85],
                       ):
    o_ds = [os.path.abspath(i) for i in output_dirs]
    for _o_d in o_ds:
        os.makedirs(_o_d,exist_ok=True)
    # Time line
    timelines = [] # suffix list [[2017_01,2017_02,....],[...],[...]]
    with open(whole_timeline_file,'rb') as f:
        whole_timeline = [i.split(':')[-1] for i in f.read().decode().split()]
        while '' in whole_timeline:
            whole_timeline.remove('')
    if dir_portion_n_list:
        time_idx_list = dir_portion_n_list
    elif dir_portion_ratio_list:
        time_idx_list = [int(len(whole_timeline)*i) for i in dir_portion_ratio_list]
    else:
        time_idx_list = [int(len(whole_timeline)*0.9)]
    time_idx_list = [0]+time_idx_list+[len(whole_timeline)]
    for _idx_ord_n, idx in enumerate(time_idx_list[:-1]):
        timelines.append(whole_timeline[idx:time_idx_list[_idx_ord_n+1]])
    
    node_fs,edge_fs,other_fs = sort_gcn_feature_dir_file(cluster_data_dir)
    
    # Split
    for _node_prfx in node_fs:
        val_arr,attrb_l = __read_node_fs__(os.path.join(cluster_data_dir,_node_prfx))
        for split_ord_idx, time_suffix_list in enumerate(timelines):
            
            _curr_attrb_l = [i for i in attrb_l if i.split(':')[-1] in time_suffix_list]
#             _curr_attrb_l_arr = np.array(_curr_attrb_l)
#             _idx_vals = [np.where(_curr_attrb_l_arr==i)[0][0] for i in _curr_attrb_l]
            _whole_attrb_l_arr = np.array(attrb_l)
            _idx_vals = [np.where(_whole_attrb_l_arr==i)[0][0] for i in _curr_attrb_l]
            _curr_val_arr = val_arr[_idx_vals,:] # TODO - Check if it is write
            
            _out_prfx=os.path.join(o_ds[split_ord_idx],_node_prfx)
            __write_node_fs__(
                val_arr=_curr_val_arr,
                attrb_l=_curr_attrb_l,
                prfx=_out_prfx)
        
    for _edge_prfx in edge_fs:
        attrb_l,_idx_dict,_weight_dict = __read_edge_fs__(os.path.join(cluster_data_dir,_edge_prfx))
        for split_ord_idx, time_suffix_list in enumerate(timelines):
            
            _curr_attrb_l_1 = [i for i in attrb_l if i.split(':')[-1] in time_suffix_list]
            _curr_idx_dict_1 = {_k:_idx_dict[_k] for _k in _curr_attrb_l_1}
            _curr_weight_dict_1 = {_k:_weight_dict[_k] for _k in _curr_attrb_l_1}

            _out_prfx=os.path.join(o_ds[split_ord_idx],_edge_prfx)
            __write_edge_fs__(
                attrb_l=_curr_attrb_l_1,
                _idx_dict=_curr_idx_dict_1,
                _weight_dict=_curr_weight_dict_1,
                prfx=_out_prfx)
        
    for _f in other_fs:
        for split_ord_idx, time_suffix_list in enumerate(timelines):
            os.system('cp %s %s'%(
                _f,
                os.path.join(o_ds[split_ord_idx],os.path.split(_f)[-1]),
            ))
    

def split_feature_time_batch(master_cluster_data_dir,
                             master_output_dirs:list,
                             dir_portion_n_list:list=[48,60],
                             dir_portion_ratio_list:list=[0.7,0.85],
                             whole_timeline_file:str='./time_line.txt',
                             multiprocess:int=8,
                             ):
    m_i_d = os.path.abspath(master_cluster_data_dir)
    m_o_ds = [os.path.abspath(i) for i in master_output_dirs]
    for i in m_o_ds:
        os.makedirs(i,exist_ok=True)
    fn_arg_list = []
    fn_kwarg_list = []
    for sub_dir in os.listdir(m_i_d):
        if os.path.isdir(os.path.join(m_i_d,sub_dir)):
            fn_arg_list.append(tuple())
            fn_kwarg_list.append(dict(
                cluster_data_dir=os.path.join(m_i_d,sub_dir),
                output_dirs=[os.path.join(i,sub_dir) for i in m_o_ds],
                dir_portion_n_list=dir_portion_n_list,
                dir_portion_ratio_list=dir_portion_ratio_list,
                whole_timeline_file=whole_timeline_file,
            ))
        
    fn = split_feature_time
    fn_args = _multi.argument_generator(fn_arg_list)
    fn_kwargs = _multi.keyword_argument_generator(fn_kwarg_list)
    _multi.multi_function_execution(
        fn=fn,
        fn_args=fn_args,
        fn_kwargs=fn_kwargs,
        max_processes=multiprocess,
        collect_result=False,
    )
    
    
def main():
    parser = argparse.ArgumentParser(description='File formatting for GCN data')
    parser.add_argument('-i','--input',help='Input directory for topics')
    parser.add_argument('-t','--timeline',help='Timeline file')
    parser.add_argument('-f','--fold_change',default=False,action='store_true',help='With FC conversion process')
    parser.add_argument('-m','--multiprocess',type=int,default=8,help='For multiprocessing N')
    parser.add_argument('-o','--output',type=str,action='append',default=[],help='Output directory suffix - in order of time index to split')
    parser.add_argument('--wo_node_fc',action='store_true',default=False,help='FC is not applied for node features')
    parser.add_argument('--wo_edge_fc',action='store_true',default=False,help='FC is not applied for edge features')
    parser.add_argument('-n','--timepoint_n',type=int,action='append',default=[],help='Specific time point  to split : int. If fold_change is imposed, n-1 would be applied')
    
    args = parser.parse_args()
    
#     input_dir = '/BiO/home/cdgu/Winery/tasting/ysu_textmining/paper/codes/topic-forecasting/_test_dump_tmp_/gcn_fc_split_test/clusters.max_structured'
#     fc_conv = '/BiO/home/cdgu/Winery/tasting/ysu_textmining/paper/codes/topic-forecasting/_test_dump_tmp_/gcn_fc_split_test/clusters.max_structured.fc_converted'
#     timeline_file='/BiO/home/cdgu/Winery/tasting/ysu_textmining/paper/codes/topic-forecasting/_test_dump_tmp_/gcn_fc_split_test/fc_time_line.txt'
    if len(args.output) == 0:
        splitting=False
    else:
        assert len(args.output) == len(args.timepoint_n)+1
        splitting=True
    input_dir = os.path.abspath(args.input)
    if args.fold_change:
        if not args.wo_node_fc and not args.wo_edge_fc:
            fc_o_dir = input_dir+'.fc_converted'
        elif not args.wo_node_fc and args.wo_edge_fc:
            fc_o_dir = input_dir+'.node_fc_converted'
        elif args.wo_node_fc and not args.wo_edge_fc:
            fc_o_dir = input_dir+'.edge_fc_converted'
        convert_fold_change_batch(
            master_cluster_data_dir=input_dir,
            master_output_dir=fc_o_dir,
            base=np.e,
            pseudocount=1.0,
            wo_node_fc=args.wo_node_fc,
            wo_edge_fc=args.wo_edge_fc,
            multiprocess=args.multiprocess,)
        if splitting:
            split_feature_time_batch(
                master_cluster_data_dir=input_dir+'.fc_converted',
                master_output_dirs=args.output,
                dir_portion_n_list=[i-1 for i in args.timepoint_n],
                dir_portion_ratio_list=[],
                whole_timeline_file=args.timeline,
                multiprocess=args.multiprocess,)
    else:
        if splitting:
            split_feature_time_batch(
                master_cluster_data_dir=input_dir,
                master_output_dirs=args.output,
                dir_portion_n_list=args.timepoint_n,
                dir_portion_ratio_list=[],
                whole_timeline_file=args.timeline,
                multiprocess=args.multiprocess,)
    
if __name__ == "__main__":
    main()
    