"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/16 10:56
@Email : handong_xu@163.com
"""
import os
import sys
import pickle
import time
from datetime import timedelta

def write_pkl_file(obj,path):
    with open(path,'wb') as ft:
        pickle.dump(obj,ft)

def read_pkl_file(path):
    with open(path,'rb') as fl:
        obj = pickle.load(fl)
    return obj
    
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))