#!/usr/bin/env python3
"""
Reconstruction of Network X object for text-mining results
Node ID: word
Node attribution : {"frquency_GROUP":freq.} Frequency could be differed by month or group
Edge: cooccurrence of two words

"""
import os
import re
import json
import itertools
import pandas as pd
import numpy as np
import pickle
import time
import operator as op
from functools import reduce
import network_utils as net_utils


#### DATA IO ####
def load_pickled(file):
    with open(file,'rb') as f:
        parsed = pickle.load(f)
    print(parsed)
    time_map = parsed.iloc[:,0].to_dict() # 
#     time_map = {k:v.split('-') for k,v in time_map.items()}
    data = parsed.iloc[:,1].to_dict()
    return data, time_map


def load_preprocessed(file:str,sep=','):
    _p_df = pd.read_csv(file,sep=sep)
    time_map = _p_df.iloc[:,0].to_dict()
#     time_map = {idx:_t.split('-') for idx,_t in time_map.items()}
    _data = _p_df.iloc[:,1].to_dict()
    data = {idx:re.sub('[]\' []','',_str).split(',') for idx,_str in _data.items()}
    return data, time_map

def save_coword_dict(target:dict,file:str):
    with open(file,'wb') as f:
        f.write(json.dumps(target).encode())
        
        
def load_coword_dict(file:str):
    with open(file,'rb') as f:
        _d = json.loads(f.read().decode())
    return _d




#################################
######## Word count dict ########
#################################

def _make_wordcount_dict_template_(word_list):
    return {i:0.0 for i in sorted(list(set(word_list)))}


# word_list: input data to count
# count_multiplication: if False, word count for a given list for all the word would be equal to each other.
def _append_word_count_(word_list:list,
                        target_word_count_dict_list:list=[],
                        count_multiplication:bool=True,
                        count_normalization:bool=False, # Normalization of word count for a list. If imposed, summation of frequency from this word_list is 1.0
                        additional_weight:float=1.0, # constant to impose weight on word count
                       ):
    if not count_multiplication:
        # consider multiple occurrence of a word as a single event
        _w_l = list(set(word_list))
    else:
        _w_l = word_list
    
    weighted_count = additional_weight
    if count_normalization:
        weighted_count = weighted_count/len(_w_l)
    for _w in _w_l:
        for _d in target_word_count_dict_list:
#             if _w not in _d:
#                 _d[_w] = 0.0
            _d[_w] += weighted_count
    return target_word_count_dict_list


def _convert_to_frequency_ratio_(frequency_dict):
    _count_sum = np.sum(list(frequency_dict.values()))
    _frq_rate_d = {k:v/_count_sum for k,v in frequency_dict.items()}
    return _frq_rate_d
    
    
def _convert_time_map_by_month_(time_map:dict):
    # 'YYYY-MM-DD' --> 'YYYY_MM'
    t_conv_f = lambda _v: f"{int(_v.split('-')[0]):04d}_{int(_v.split('-')[1]):02d}" if type(_v)==str \
        else f"{_v.year:04d}_{_v.month:02d}"
    by_month_time_map = {_k:t_conv_f(_v) for _k, _v in time_map.items()}
    return by_month_time_map
    
    
def _make_serial_month_times_(min_yr:int,min_mon:int,max_yr:int,max_mon:int):
    min_yr = int(min_yr)
    min_mon = int(min_mon)
    max_yr = int(max_yr)
    max_mon = int(max_mon)
    
    times = []
    for _yr in range(min_yr,max_yr+1):
        for _mon in range(1,13):
            if _yr == min_yr and _mon < min_mon:
                continue
            if _yr == max_yr and _mon > max_mon:
                continue
            times.append(f"{_yr:04d}_{_mon:02d}")
    return times

    
# preprocessed_data_dict: {KEY:WORD_LIST} data shape supposed to be document-wise word list
# count_multiplication: if False, multiplicated word for a given word list would be counted equally.
# timeline_normalization: contribution of a timeline to whole word count would be normalized.
# document_normalization: contribution of a document to a timeline would be normalized.
# that is, if imposed, word count for a document would be normalized to 1.0. (Summation of frequency for basal-level of word_list is 1.0)
# sentence_normalization: contribution of sentence to a document would be normalized.
# Normalization of word count by given list. If imposed, summation of frequency from this word_list is 1.0
def get_wordcount_dict(preprocessed_data_dict:dict, #{IDX:[WORD]}
                       time_map:dict, # {IDX:YYYY_MM}
                       fill_empty_time:bool=True, # only month-level time line is supportable
                       input_at_document_level:bool=True,
                       count_multiplication:bool=True,
                       timeline_normalization:bool=False,
                       document_normalization:bool=False,
                       sentence_normalization:bool=False,
                       low_mem_mode:bool=False,
                       low_mem_chunk_dir:str='./by_month_word_chunk',
                      )->dict:
    # Getting word_list
    word_list = []
    if input_at_document_level:
        for _k, _each_wl in preprocessed_data_dict.items():
            word_list.extend(_each_wl)
    else:
        for _k, _sent_l in preprocessed_data_dict.items():
            for _each_wl in _sent_l:
                word_list.extend(_each_wl)
    word_list = sorted(list(set(word_list)))
    
    times = sorted(list(set(time_map.values())))
    # Fill empty time stamp
    if fill_empty_time: # only month-level
        [min_yr, min_mon] = times[0].split('_')
        [max_yr, max_mon] = times[-1].split('_')
        times = _make_serial_month_times_(
            min_yr=min_yr,min_mon=min_mon,
            max_yr=max_yr,max_mon=max_mon
        )
        
    # Word count dict
    _tmpl_ = _make_wordcount_dict_template_(word_list=word_list)
    whole_word_count = _tmpl_.copy()
    if low_mem_mode:
        by_month_word_count = {_t:os.path.join(low_mem_chunk_dir,f"{_t}.word_count.json") for _t in times}
        for _t, _f in by_month_word_count.items():
            net_utils.write_json(data=_tmpl_,file=_f)
    else:
        by_month_word_count = {_t:_tmpl_.copy() for _t in times}

    # Getting word count
    if input_at_document_level: # preprocessed_data_dict: {KEY:WORD_LIST}
        for _idx, curr_word_list in preprocessed_data_dict.items():
            curr_time = time_map[_idx]
            if low_mem_mode:
                curr_d = net_utils.parse_json(file=by_month_word_count[curr_time])
                [curr_d] = _append_word_count_(
                    word_list=curr_word_list,
                    target_word_count_dict_list=[curr_d],
                    count_multiplication=count_multiplication,
                    count_normalization=document_normalization,
                    additional_weight=1.0,
                )
                net_utils.write_json(data=curr_d,file=by_month_word_count[curr_time])
            else:
                [by_month_word_count[curr_time]] = _append_word_count_(
                    word_list=curr_word_list,
                    target_word_count_dict_list=[by_month_word_count[curr_time]],
                    count_multiplication=count_multiplication,
                    count_normalization=document_normalization,
                    additional_weight=1.0,
                )
        
    else: # preprocessed_data_dict: {KEY:[WORD_LIST]]}
        for _idx, sents in  preprocessed_data_dict.items():
            curr_time = time_map[_idx]
            # Normalization constant: for document_normalization and not sentence_normalization
            if document_normalization and not sentence_normalization:
                if count_multiplication:
                    curr_idx_word_count_sum = np.sum([len(i) for i in sents])
                else:
                    curr_idx_words = []
                    for _sent in sents:
                        curr_idx_words.extend(_sent)
                    curr_idx_words = set(curr_idx_words)
                    curr_idx_word_count_sum = len(curr_idx_words)
                
            for curr_word_list in sents:
                if document_normalization:
                    if sentence_normalization:
                        additional_weight = 1.0/len(sents)
                    else:
                        additional_weight = 1.0/curr_idx_word_count_sum
                else:
                    additional_weight = 1.0
                if low_mem_mode:
                    curr_d = net_utils.parse_json(file=by_month_word_count[curr_time])
                    [curr_d] = _append_word_count_(
                        word_list=curr_word_list,
                        target_word_count_dict_list=[curr_d],
                        count_multiplication=count_multiplication,
                        count_normalization=document_normalization,
                        additional_weight=additional_weight,
                    )
                    net_utils.write_json(data=curr_d,file=by_month_word_count[curr_time])
                else:
                    [by_month_word_count[curr_time]] = _append_word_count_(
                        word_list=curr_word_list,
                        target_word_count_dict_list=[by_month_word_count[curr_time]],
                        count_multiplication=count_multiplication,
                        count_normalization=sentence_normalization,
                        additional_weight=additional_weight,
                        )
    
    if timeline_normalization: # frequency rate
        for _t in times:
            _wc_d = by_month_word_count[_t]
            if low_mem_mode:
                _wc_d = net_utils.parse_json(file=by_month_word_count[_t])
            _timeline_sum = np.sum(list(_wc_d.values()))
            _wc_d_l = list(_wc_d)
            for _w in _wc_d_l:
                _wc_d[_w] = _wc_d[_w]/_timeline_sum
            if low_mem_mode:
                net_utils.write_json(data=_wc_d,file=by_month_word_count[_t])
                del _wc_d
                
    for _t in times:
        _wc_d = by_month_word_count[_t]
        if low_mem_mode:
            _wc_d = net_utils.parse_json(file=by_month_word_count[_t])
        for _w, _cnt in _wc_d.items():
            whole_word_count[_w] += _cnt
        del _wc_d

    return word_list, whole_word_count, by_month_word_count



# Dropping words with 
def drop_lowcount_words(word_list:list,
                        whole_word_count:dict,
                        by_month_word_count:dict,
                        word_count_limit:int=0,
                       ):
    if not word_count_limit:
        return word_list, whole_word_count, by_month_word_count
    
    _cnt_ser = pd.Series(whole_word_count,index=word_list)
    _cnt_ser = _cnt_ser.loc[_cnt_ser>word_count_limit]
    ref_word_list = list(_cnt_ser.index)
    # by_month_word_count
    whole_times = list(by_month_word_count.keys())
    for _time in whole_times:
#         by_month_word_count[_time] = {
#             _k:_v for _k, _v in by_month_word_count[_time].items() if _k in ref_word_list}
        by_month_word_count[_time] = pd.Series(
            by_month_word_count[_time],
            index=list(set(by_month_word_count[_time].keys())&set(ref_word_list))
        ).to_dict()
    return ref_word_list, _cnt_ser.to_dict(), by_month_word_count


#################################
########## Coword dict ##########
#################################

# make coword dict template using word list
def _make_coword_dict_template_(word_list:list):
    _d = {tuple(sorted(list(_k))):0 for _k in itertools.combinations(
        sorted(list(set(word_list))),2)}
    return _d
    

# count cooccurrence of word and append to collectin dictionary
# target_coword_dict_list : coword collection dicts in list. multiple dictionaries can be appllied
def _append_coword_(word_list:list,
                    target_coword_dict_list:list=[],
                    count_multiplication:bool=True,
                    count_normalization:bool=False, # Normalization of cooccurence count. If imposed, summation of cooccurrence from this word_list is 1.0
                    additional_weight:float=1.0, # constant to impose weight on cooccurrence count
                   ):
    if not count_multiplication:
        # consider multiple occurrence of a word as a single event
        _w_l = list(set(word_list))
    else:
        _w_l = word_list
        
    _ws = []
    for _w_k in itertools.combinations(_w_l,2):
        _w_k = tuple(sorted(list(_w_k))) # NOTE: (WORD1, WORD2) with sorted order
        if _w_k[0] != _w_k[1]:
            _ws.append(_w_k)
    weighted_count = additional_weight
    if count_normalization:
        weighted_count = weighted_count/len(_ws)
    for _w_k in _ws:
        for _d in target_coword_dict_list:
            if _w_k not in _d:
                _d[_w_k] = 0.0
            _d[_w_k] += weighted_count
    return target_coword_dict_list
    
# nCc = n!/c!(n-c)!
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom
    
    
def make_coword_dict(preprocessed_data_dict:dict, #{IDX:[WORD]}
                     whole_word_list:list,
                     time_map:dict, # {IDX:YYYY_MM}
                     fill_empty_time:bool=True, # only month-level time line is supportable
                     input_at_document_level:bool=True,
                     
                     count_multiplication:bool=True,
                     timeline_normalization:bool=False,
                     document_normalization:bool=False,
                     sentence_normalization:bool=False,
                     low_mem_mode:bool=False,
                     low_mem_chunk_dir:str='./by_month_coword_chunk',
                    )->dict:
    
    # reorganize data by time
    times = sorted(list(set(time_map.values())))
    # Fill empty time stamp
    if fill_empty_time: # only month-level
        [min_yr, min_mon] = times[0].split('_')
        [max_yr, max_mon] = times[-1].split('_')
        times = _make_serial_month_times_(
            min_yr=min_yr,min_mon=min_mon,
            max_yr=max_yr,max_mon=max_mon
        )
    # Make time_to_idx dict: {TIME:[IDX]}
    time_to_idx = {_t:[] for _t in times}
    for _idx, _time in time_map.items():
        if _time not in time_to_idx:
            time_to_idx[_time] = []
        time_to_idx[_time].append(_idx)
    
    whole_coword_dict = _make_coword_dict_template_(whole_word_list)
    if low_mem_mode:
        by_month_coword_dict = {_t:os.path.join(low_mem_chunk_dir,f"{_t}.coword.json") for _t in times}
        for _t, _cowrd_f in by_month_coword_dict.items():
            net_utils.write_lowmem_coword_dict(data={},file=_cowrd_f)
    else:
        by_month_coword_dict = {_t:{} for _t in times}
    # 
    if input_at_document_level:
        for _k, curr_word_list in preprocessed_data_dict.items():
            curr_word_list = [i for i in curr_word_list if i in whole_word_list]
            curr_time = time_map[_k]
            if low_mem_mode:
                curr_d = net_utils.parse_lowmem_coword_dict(file=by_month_coword_dict[curr_time])
                [curr_d] =_append_coword_(
                    word_list=curr_word_list,
                    target_coword_dict_list=[curr_d],
                    count_multiplication=count_multiplication,
                    count_normalization=document_normalization,
                    additional_weight=1.0,
                )
                net_utils.write_lowmem_coword_dict(data=curr_d,file=by_month_coword_dict[curr_time])
            else:
                [by_month_coword_dict[curr_time]]=_append_coword_(
                    word_list=curr_word_list,
                    target_coword_dict_list=[by_month_coword_dict[curr_time]],
                    count_multiplication=count_multiplication,
                    count_normalization=document_normalization,
                    additional_weight=1.0,
                )
    else: # input data would be {IDX:[[WORDS]]}
        for _k, sents in preprocessed_data_dict.items():
            curr_time = time_map[_k]
            # Normalization constant: for document_normalization and not sentence_normalization
            if document_normalization and not sentence_normalization:
                curr_idx_words = []
                for _sent in sents:
                    curr_idx_words.extend(_sent)
                curr_idx_words_c_d = {i:curr_idx_words.count(i) for i in set(curr_idx_words) if i in whole_word_list}
                curr_idx_coword_sum = ncr(n=len(curr_idx_words_c_d.keys()),r=2)
                
                if count_multiplication:
                    for _wrd_,_wrd_cnt in curr_idx_words_c_d.items():
                        if _wrd_cnt != 1:
                            curr_idx_coword_sum = curr_idx_coword_sum-ncr(_wrd_cnt,2)
            # 
            for curr_word_list in sents:
                if document_normalization:
                    if sentence_normalization:
                        additional_weight = 1.0/len(sents)
                    else:
                        additional_weight = 1.0/curr_idx_coword_sum
                else:
                    additional_weight=1.0
                if low_mem_mode:
                    curr_d = net_utils.parse_lowmem_coword_dict(file=by_month_coword_dict[curr_time])
                    [curr_d] =_append_coword_(
                        word_list=curr_word_list,
                        target_coword_dict_list=[curr_d],
                        count_multiplication=count_multiplication,
                        count_normalization=document_normalization,
                        additional_weight=additional_weight,
                    )
                    net_utils.write_lowmem_coword_dict(data=curr_d,file=by_month_coword_dict[curr_time])
                else:
                    [by_month_coword_dict[curr_time]]=_append_coword_(
                        word_list=curr_word_list,
                        target_coword_dict_list=[by_month_coword_dict[curr_time]],
                        count_multiplication=count_multiplication,
                        count_normalization=sentence_normalization,
                        additional_weight=additional_weight,
                    )
                
    # Normalization(optional)            
    if timeline_normalization: # convert to cooccurrence count to rate
        for _t in times:
            _cocr_c_d = by_month_coword_dict[_t]
            if low_mem_mode:
                _cocr_c_d = net_utils.parse_lowmem_coword_dict(file=by_month_coword_dict[_t])
            _timeline_sum = np.sum(list(_cocr_c_d.values()))
            _cocr_c_d_l = list(_cocr_c_d.keys())
            for _cocr_pair in _cocr_c_d_l:
                _cocr_c_d[_cocr_pair] = _cocr_c_d[_cocr_pair]/_timeline_sum
            if low_mem_mode:
                net_utils.write_lowmem_coword_dict(data=_cocr_c_d,file=by_month_coword_dict[_t])
                del _cocr_c_d
    
    for _t in times:
        _cocr_c_d = by_month_coword_dict[_t]
        if low_mem_mode:
            _cocr_c_d = net_utils.parse_lowmem_coword_dict(file=by_month_coword_dict[_t])
        for _cocr_pair, _cnt in _cocr_c_d.items():
#             if _cocr_pair in whole_coword_dict:
            whole_coword_dict[_cocr_pair] += _cnt
#             # word_list doesn't have a word in _cocr_pair: in case of word count limit
#             else:
#                 pass
        if low_mem_mode:
            del _cocr_c_d
    
    return whole_coword_dict, by_month_coword_dict
    
    
#################################
######### Main function #########
#################################


def parse_preprocessed_data(data_file:str,
                            output_dir:str=False,
                            output_file:str=False,
                            fill_empty_time=True,
                            input_at_document_level=True,
                            count_multiplication:bool=True,
                            timeline_normalization:bool=False,
                            document_normalization:bool=False,
                            sentence_normalization:bool=False,
                            low_mem_mode:bool=False,
                            word_count_limit:int=0,
                            without_coword:bool=False,
                           ):
    data, time_map = load_pickled(data_file)
    time_map_by_month = _convert_time_map_by_month_(time_map)
    if output_dir:
        os.makedirs(output_dir,exist_ok=True)
        
    if low_mem_mode:
        if not output_dir:
            output_dir=os.path.abspath('./')
        by_month_word_chunk_dir = os.path.join(output_dir,'by_month_word_chunk')
        os.makedirs(by_month_word_chunk_dir,exist_ok=True)
        word_list_file = os.path.join(output_dir,'word_list.tsv')
        whole_word_count_file = os.path.join(output_dir,'whole_word_count.json')
        by_month_word_count_file = os.path.join(output_dir,'by_month_word_count_file.json')
        
        by_month_coword_chunk_dir = os.path.join(output_dir,'by_month_coword_chunk')
        os.makedirs(by_month_coword_chunk_dir,exist_ok=True)
        whole_coword_map_file = os.path.join(output_dir,'whole_coword_map.json')
        by_month_coword_file = os.path.join(output_dir,'by_month_coword_file.json')
    else:
        by_month_word_chunk_dir = None
        by_month_coword_chunk_dir = None
        
    print('Getting word count dictionary...')
    word_list, whole_word_count, by_month_word_count = get_wordcount_dict(
        preprocessed_data_dict=data,
        time_map=time_map_by_month,
        fill_empty_time=fill_empty_time,
        input_at_document_level=input_at_document_level,
        count_multiplication=count_multiplication,
        timeline_normalization=timeline_normalization,
        document_normalization=document_normalization,
        sentence_normalization=sentence_normalization,
        low_mem_mode=low_mem_mode,
        low_mem_chunk_dir=by_month_word_chunk_dir,
    )
    print('Word count limit:\t%s'%word_count_limit)
    word_list, whole_word_count, by_month_word_count = drop_lowcount_words(
        word_list=word_list,
        whole_word_count=whole_word_count,
        by_month_word_count=by_month_word_count,
        word_count_limit=word_count_limit)
    
    if low_mem_mode:
        net_utils.write_list(data=word_list,file=word_list_file)
        net_utils.write_json(data=whole_word_count,file=whole_word_count_file)
        net_utils.write_json(data=by_month_word_count,file=by_month_word_count_file)
        del whole_word_count
    
    if without_coword:
        if output_dir:
            with open(os.path.join(output_dir,'coword_results.pkl'),'wb') as f:
                pickle.dump((word_list, whole_word_count, by_month_word_count),f)
        elif output_file:
            with open(output_file,'wb') as f:
                pickle.dump((word_list, whole_word_count, by_month_word_count),f)
        return word_list, whole_word_count, by_month_word_count
        
    else:
        print('Getting coword dictionary...')
        whole_coword_map, by_month_coword_map = make_coword_dict(
            preprocessed_data_dict=data,
            whole_word_list=word_list,
            time_map=time_map_by_month,
            fill_empty_time=fill_empty_time,
            input_at_document_level=input_at_document_level,
            count_multiplication=count_multiplication,
            timeline_normalization=timeline_normalization,
            document_normalization=document_normalization,
            sentence_normalization=sentence_normalization,
            low_mem_mode=low_mem_mode,
            low_mem_chunk_dir=by_month_coword_chunk_dir,
        )

        if low_mem_mode:
            net_utils.write_lowmem_coword_dict(data=whole_coword_map,file=whole_coword_map_file)
            del whole_coword_map
            net_utils.write_json(data=by_month_coword_map,file=by_month_coword_file)
            return word_list_file, whole_word_count_file, by_month_word_chunk_dir, whole_coword_map_file, by_month_coword_chunk_dir

        else:
            if output_dir:
                with open(os.path.join(output_dir,'coword_results.pkl'),'wb') as f:
                    pickle.dump((word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map),f)
            elif output_file:
                with open(output_file,'wb') as f:
                    pickle.dump((word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map),f)
            return word_list, whole_word_count, by_month_word_count, whole_coword_map, by_month_coword_map

        