import os

import torch
import time
import numpy as np
from module.dataloader import create_data_loader,read_file,build_vocab,build_dataset
import argparse
from importlib import import_module
from transformers import BertTokenizer
from module.log import logger
from module.train import train,init_network
# from module.utils import write_pkl_file,read_pkl_file

# 设置随机数种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser(description="Chinese Text Classification")
    parser.add_argument("--activate", type=str, default=False, help="choose an activate function,default is Mish")
    parser.add_argument('--embedding', default="random", type=str, help="random or pretrain")
    parser.add_argument('--optim', default=False, help="choose an optim function:[SGD,Adagrad,RMSProp,Adadelta,Adam]")
    parser.add_argument('--init', default=False, type=str, help="choose a method for init model:[xavier,kaiming]")
    
    args = parser.parse_args()

    embedding = args.embedding

    x = import_module("models."+"TextCNN")
    config = x.Config("Student",embedding)
    start_time = time.time()
    logger.info("Train iters is Done ...")

    vocab,df_train,df_val,df_test = build_dataset(config)
    trainloader = create_data_loader(df_train,vocab,config)
    validloader = create_data_loader(df_val,vocab,config)
    testloader = create_data_loader(df_test,vocab,config)
    logger.info("Data loader is done")

    config.n_vocab = len(vocab)
    print(config.n_vocab)
    config.init_method = args.init
    config.activate = args.activate
    config.optimizer = args.optim
    model = x.Model(config).to(config.device)
    logger.info("train in {} device".format(config.device))
    logger.info('start initial network parameters')
    init_network(model,config)
    logger.info('finish initial network parameters')

    train(config, model, trainloader, validloader, testloader)




