"""
本代码对的功能：
    1、初始化模型
    2、模型的训练、测试等
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import *
from logger import logger


class Model():
    def __init__(self,num_class):
        #
        self.param = {
            'objective': 'multiclass',
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss',
            'num_class': num_class,
            'num_leaves': 16,
            'max_bin': 255,
            'max_depth': -1,
            "learning_rate": 0.2,
            "colsample_bytree": 0.8,  # 每次迭代中随机选择特征的比例
            "bagging_fraction": 0.8,  # 每次迭代时用的数据比例
            'bagging_freq': 5,  # 防止过拟合
            'feature_fraction': 0.8,  # 防止过拟合
            'min_child_samples': 25,
            'n_jobs': -1,
            'seed': 1000,
            # 'min_data_in_leaf': 21,  # 防止过拟合
            # 'min_sum_hessian_in_leaf': 3.0,  # 防止过拟合
        }  #设置出参数



        self.num_boost_round = 3000000
        print(f"num class is {num_class}")

    def train(self,xtrain,ytrain,xvalid,yvalid):
    
        lgb_train = lgb.Dataset(xtrain, ytrain)
        lgb_valid = lgb.Dataset(xvalid, yvalid,reference=lgb_train)
        gbm = lgb.train(self.param,
                lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_train, lgb_valid])
        return gbm

    def test(self, xtest, clf):
        pred = clf.predict(xtest,predict_disable_shape_check=True)
        print(pred)
        y_predict = np.argmax(pred, axis=1)
        return y_predict

    def evaluate(self,label,predict,tags=None):
        label = np.array(label).astype(int)
        predict = np.array(predict).astype(int)
        precision_res = precision_score(label,predict,average='micro')
        recall_res = recall_score(label,predict,average='micro')
        f1_res = f1_score(label,predict, average='micro')
        report = classification_report(label, predict, digits=4,target_names=tags)
        logger.info(f"report is {report}")
        return precision_res,recall_res,f1_res,report