"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/27 11:10
@Email : handong_xu@163.com
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from importlib import import_module
from module.Ranger import opt_func
from tensorboardX import SummaryWriter
# from module.dataloader import build_dataset,build_iterator,get_time_dif
from module.utils import get_time_dif
from transformers import BertModel, BertTokenizer, AdamW
import argparse


# 权重初始化，默认xavier
def init_network(model, config, exclude='embedding', seed=123):
    print('init model......')
    print(config.init_method)
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if config.init_method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif config.init_method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config,model,train_iter,dev_iter,test_iter):
    start_time = time.time()
    model.train()
    print("optimizer:{}".format(config.optimizer))
    if config.model_name == "Bert":
        optim = AdamW(model.parameters(),lr=0.5,weight_decay=0.02)
    elif not config.optimizer:
        optim = opt_func(model.parameters(),lr=config.learning_rate)
    elif config.optimizer == 'SGD':
        if not config.moment:
            optim = torch.optim.SGD(model.parameters(),lr=config.learning_rate)
        else:
            optim = torch.optim.SGD(model.parameters(),lr=config.learning_rate,momentum=config.moment)
    elif config.optimizer == 'Adagrad':
        optim = torch.optim.Adagrad(model.parameters(),lr=config.learning_rate)
    elif config.optimizer == 'RMSprop':
        optim = torch.optim.RMSprop(model.parameters(),lr=config.learning_rate,alpha=config.alpha)
    elif config.optimizer == 'Adadelta':
        optim = torch.optim.Adadelta(model.parameters(),rho=0.9)
    elif config.optimizer == 'Adam':
        optim = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    total_batch = 0 # 记录进行了多少个batch
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epoches):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epoches))
        # scheduler.step() # 学习率衰减
        for i, unit in enumerate(train_iter):
            if config.model_name == "TextCNN":
                texts = unit['embed']
                labels = unit['labels']
                outputs = model(texts)
            else:
                input_ids = unit["input_ids"]
                attention_mask = unit["attention_mask"]
                labels = unit["labels"]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optim.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path+'/checkpoint{}.pth'.format('best'))
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path+'/checkpoint{}.pth'.format('best')))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for i, unit in enumerate(data_iter):
            if config.model_name == "TextCNN":
                texts = unit['embed']
                labels = unit['labels']
                outputs = model(texts)
            else:
                input_ids = unit["input_ids"]
                attention_mask = unit["attention_mask"]
                labels = unit["labels"]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            # print(outputs)
            # predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            # print(labels)
            # s = input()
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        target_names = ['教育','家居','时尚','时政','科技','房产','财经']
        report = metrics.classification_report(labels_all, predict_all, target_names=target_names, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--activate",type=str,default=False,help="choose an activate function,default is Mish")
parser.add_argument('--embedding',default="random",type=str,help="random or pretrain")
parser.add_argument('--optim',default=False,help="choose an optim function:[SGD,Adagrad,RMSProp,Adadelta,Adam]")
parser.add_argument('--init',default=False,type=str,help="choose a method for init model:[xavier,kaiming]")
args = parser.parse_args()

if __name__ == '__main__':
    import os
    from utils.utils import *
    from utils.log import logger
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True  #

    embedding = args.embedding
    x = import_module('module.' + 'TextCNN')
    config = x.Config(DATA_DIR,embedding)
    start_time = time.time()
    logger.info("Loading data...")
    file_dir = os.path.join(DATA_DIR, 'labeled_data.csv')
    vocab, train_data, dev_data, test_data,classes = build_dataset(file_dir)
    logger.info("Finish Loading data...")
    logger.info("Build train iters...")
    train_iter = build_iterator(train_data,config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data,config)
    logger.info("Train iters is Done ...")
    config.n_vocab = len(vocab)
    print(config.n_vocab)
    config.init_method = args.init
    config.class_list = classes
    config.activate = args.activate
    config.optimizer = args.optim
    model = x.Model(config).to(config.device)
    logger.info("train in {} device".format(config.device))
    logger.info('start initial network parameters')
    init_network(model,config)


    train(config, model, train_iter, dev_iter, test_iter)




