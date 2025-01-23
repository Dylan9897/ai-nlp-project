"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 14:37
@Email : handong_xu@163.com
"""
import argparse
from module.config import Config



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CWS")
    parser.add_argument('--model',default='prefix',type=str,help='choose a model:[prefix or hmm]')
    args = parser.parse_args()
    config = Config()
    model = args.model
    config.model_name = model


    if model == "prefix":
        from trainer.train_prefix import train
    elif model == 'hmm':
        from trainer.train_hmm import train

    else:
        pass
    train(config)





