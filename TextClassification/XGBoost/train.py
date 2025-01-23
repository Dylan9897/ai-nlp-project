"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/4/19 17:56
@Email : handong_xu@163.com
"""
import argparse
from models.XGBoost import Model
from data_process import DataProcessor,Converter,dump
from logger import logger


def main(args):


    train_path = "data/train.xlsx"
    valid_path = "data/test.xlsx"
    data_processor = DataProcessor(args)
    converter = Converter()

    df_train = data_processor._read_file(train_path)
    df_valid = data_processor._read_file(valid_path)

    logger.info(f"Numbers of trainset is {len(df_train)}")
    logger.info(f"Numbers of validset is {len(df_valid)}")

    label_dic = data_processor._return_label_dic(df_train)
    num_classes = len(label_dic)
    logger.info(f"Number of class is {label_dic}")

    # 读取数据集
    x_train, y_train = data_processor._return_dataset(df_train, label_dic)
    x_valid, y_valid = data_processor._return_dataset(df_valid, label_dic)

    # 向量化
    xtrain = converter.tfvectorize(x_train)
    xvalid = converter.tfvectorize(x_valid, test=True)

    model = Model(num_classes)

    clf = model.train(xtrain, y_train, xvalid, y_valid)
    dump('ckpt/xgboost.trans_base.model',clf)
    pred = model.test(xvalid,clf)
    model.evaluate(y_valid,pred)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument('--cws',default=True,help='cut words or not')
    args = parser.parse_args()
    main(args)
