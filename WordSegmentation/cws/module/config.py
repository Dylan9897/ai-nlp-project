"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 14:44
@Email : handong_xu@163.com
"""
"""
所有的参数都在这里
"""

class Config():
    def __init__(self):
        self.model_name = ''
        self.train_file = 'data/pku/train/pku_training.utf8'
        self.test_file = 'data/pku/test/pku_test.utf8'
        self.vocab_file = 'data/pku/prefix/vocab.txt'
        self.hmm_train_file = 'data/pku/train/pku_hmm_training.utf8'
