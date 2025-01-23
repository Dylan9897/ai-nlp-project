"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 14:41
@Email : handong_xu@163.com
"""
"""
基于hmm模型进行分词
"""
import os
from module.log import logger
from module.dataloader import read_file,trans_hmm
from models.hmm import HMM


def train(config):
    if not os.path.exists(config.hmm_train_file):
        logger.info("merging hmm data")
        fileList = read_file(config.train_file)
        trans_hmm(fileList, config.hmm_train_file)
    model = HMM(config.train_file,config.hmm_train_file)
    model.train()

