import os

import torch
import numpy as np
import argparse
import sys
sys.path.append(os.getcwd())

from TextCNN.module import TextCNN
from TextCNN.module.data_loader import build_dataset,create_data_loader
from TextCNN.utils.logger import logger
from TextCNN.module.trainer import train

# 设置随机数种子
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser(description="Chinese Text Classification")
    parser.add_argument("--dataset",type=str,default="longnews",help="choose an datasets")
    parser.add_argument("--cws", type=bool, default=True, help="cws or not")
    parser.add_argument("--activate", type=str, default=False, help="choose an activate function,default is Mish")
    parser.add_argument('--embedding', default="random", type=str, help="random or pretrain")
    parser.add_argument('--optim', default=False, help="choose an optim function:[SGD,Adagrad,RMSProp,Adadelta,Adam]")
    parser.add_argument('--init', default=False, type=str, help="choose a method for init model:[xavier,kaiming]")
    args = parser.parse_args()
    config = TextCNN.Config(args)

    vocab, df_train, df_val, df_test = build_dataset(config)
    print(len(vocab),len(df_train),len(df_val),len(df_test))
    logger.info("Build dataset is done")
    trainloader = create_data_loader(df_train, vocab, config)
    validloader = create_data_loader(df_val, vocab, config)
    if not df_test:
        testloader = validloader
    else:
        testloader = create_data_loader(df_test, vocab, config)
    logger.info("Data loader is done")

    config.n_vocab = len(vocab)
    config.init_method = args.init
    config.activate = args.activate
    config.optimizer = args.optim

    model = TextCNN.Model(config).to(config.device)
    print(model)

    train(config, model, trainloader, validloader, testloader)






