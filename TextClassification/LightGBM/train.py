import argparse
from logger import logger
from data_process import DataProcessor,Converter,dump
from model import Model
import numpy as np


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
    x_train,y_train = data_processor._return_dataset(df_train,label_dic)
    x_valid,y_valid = data_processor._return_dataset(df_valid,label_dic)

    # 向量化
    xtrain = converter.tfvectorize(x_train)
    xvalid = converter.tfvectorize(x_valid,test=True)
    
    # 实例化模型
    model = Model(num_classes)
    
    logger.info("start training")
    clf = model.train(xtrain,y_train, xvalid,y_valid)
    
    predict = model.test(xvalid,clf)

    logger.info(f"preiction is {predict}")

    model.evaluate(y_valid,predict,tags = list(label_dic.keys()))
    dump('ckpt/lightgbm.trans_{}.model'.format(args.cws), clf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument('--cws',default=True,help='cut words or not')
    args = parser.parse_args()
    main(args)


