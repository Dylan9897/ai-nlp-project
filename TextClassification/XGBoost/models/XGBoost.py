"""
-*- coding: utf-8 -*-
@Author : dongdong
@Time : 2022/4/19 20:16
@Email : handong_xu@163.com
"""

import xgboost as xgb
import numpy as np
from sklearn.metrics import *
from utils.converter import Converter
import matplotlib.pyplot as plt

class Model():
    def __init__(self,num_class):
        self.param = {
        'objective':'multi:softmax',
        # 'objective': 'multi:softprob',
        'num_class':num_class,
        'eta':0.1,
        'booster':'gbtree',
        'max_depth':200,
        # 'verbosity':3,
        'silent':5,
        'nthread':4
        }
        self.num_round = 500
        self.convert = Converter()

    def train(self,xtrain,ytrain,xvalid,yvalid):
        xg_train = xgb.DMatrix(xtrain, label=ytrain)
        xg_valid = xgb.DMatrix(xvalid, label=yvalid)
        dataList = [(xg_train, 'train'),(xg_valid,'valid')]
        bst = xgb.train(self.param,  # 参数
                        xg_train,  # 训练数据
                        self.num_round,  # 弱学习器的个数
                        dataList)
        return bst

    def test(self,xtest,clf):
        xg_test = xgb.DMatrix(xtest)
        pred = clf.predict(xg_test)
        return pred

    def evaluate(self,label,predict):
        label = np.array(label).astype(int)
        predict = np.array(predict).astype(int)
        precision_res = precision_score(label,predict,average='micro')
        recall_res = recall_score(label,predict,average='micro')
        f1_res = f1_score(label,predict, average='micro')
        report = classification_report(label, predict, digits=4)
        print(report)
        return precision_res,recall_res,f1_res,report

    def print_roc(self,label,predict):
        label = np.array(label).reshape(-1,1).astype(int)
        predict = np.array(predict).reshape(-1,1).astype(int)
        lr_auc = roc_auc_score(label, predict,multi_class='ovr')
        lr_fpr, lr_tpr, lr_threasholds = roc_curve(label, predict)  # 计算ROC的值,lr_threasholds为阈值
        plt.title("roc_curve of %s(AUC=%.4f)" % ('logist', lr_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(lr_fpr, lr_tpr)
        plt.savefig('conf/roc.jpg')












