"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 14:41
@Email : handong_xu@163.com
"""
"""
基于前缀字典进行分词
"""
from module.log import logger
from module.dataloader import merge_prefix_vocab



def train(config):
    logger.info("start to merge prefix vocab...")
    merge_prefix_vocab(config.train_file,config.vocab_file)
    logger.info("finish merging prefix vocab...")