#!/usr/bin/env python3
'''
Module for Multiprocessing and multithreading methods
Developed and wrapped with python 3 by A.R. C. Gu, Oncocross, Co., Ltd.
'''
import multiprocessing
import concurrent.futures
import time
from typing import Optional, Iterator, Callable

MAX_CPU = multiprocessing.cpu_count()

# Multi process for series of executions for a function
def multi_function_execution(fn,
                             fn_args,
                             fn_kwargs,
                             max_processes=1,
                             collect_result=True,
                             ):
    start = time.time()
    print('Multiprocessing with _multi module')
    print('Max workers: %s'%(max_processes))
    """Execute the given callable concurrently using multiple threads and/or processes."""
    # Ref: https://stackoverflow.com/a/57999709/
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_processes)  # type: ignore

    futures = []
    while True:
        try:
            fn_args_cur = next(fn_args)
            fn_kwargs_cur = next(fn_kwargs)
            future = executor.submit(fn, *fn_args_cur, **fn_kwargs_cur)
            futures.append(future)
        except StopIteration:
            break
        except Exception as e:
            raise(e)
    results=[]
    
    # Task counting for reporting
    task_no = len(futures)
    c = 0
    time_span = int(max_processes)
    if time_span >= task_no:
        time_span = int(task_no/2.0)
    if time_span < 1:
        time_span = 1
    # Task run
    for future in concurrent.futures.as_completed(futures):
        c += 1
        if c%time_span == 0:
            curr_time = time.time()
            print('MULTI_PROCESSING_INFO: %s %% proceeded in %s secs'%(c/task_no*100.0, curr_time-start))
        if collect_result:
            results.append(future.result())
        else:
            results = future.result()

    end = time.time()
    print('Task finished with %s workers in %s seconds' %(
        max_processes,end-start
        ))
    print('Task %s finished with %s workers in %s seconds' %(
        fn.__name__,max_processes,end-start
        ))
    return results


# Argument generator for multiprocessing
# args_list: list (or iterative) of arguments: [(args_process1),(args_process2),..]
def argument_generator(args_list):
    for _args in args_list:
        yield _args
        
        
# Keyword argument generator for multiprocessing
# kwargs_list: list (or iterative) of arguments: [{kwargs_process1},{args_process2},..]
def keyword_argument_generator(kwargs_list,constant_kwargs={}):
    for _kwargs in kwargs_list:
        if constant_kwargs:
            _kwargs.update(constant_kwargs)
        yield _kwargs
