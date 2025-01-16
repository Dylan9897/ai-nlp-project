import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from TextCNN.module.Mish import Mish

class Config(object):
    """配置参数"""
    def __init__(self,args,kwargs=None):
        self.args = args
        if not kwargs:
            kwargs = {}
        self.cws = args.cws
        self.model_name = "TextCNN"
        self.train_path = "data/datasets/{}/train.json".format(args.dataset)
        self.dev_path = "data/datasets/{}/dev.json".format(args.dataset)
        self.test_path = "data/datasets/{}/test.json".format(args.dataset)
        self.vocab_path = "data/datasets/{}/vocab.pkl".format(args.dataset)
        self.class_list = self.get_classlist()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.save_path = "output" if os.path.exists("output") else os.mkdir("output")
        self.embedding_pretrained = torch.tensor(
            np.load('ckpt/' + args.embedding)["embeddings"].astype('float32')) \
            if args.embedding != 'random' else None  # 预训练词向量

        self.embed_dims = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300  # 字向量维度

        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.dropout = kwargs.get("dropout",0.5)
        self.require_improvement = kwargs.get("dropout",10000)  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epoches = kwargs.get("num_epoches",20)
        self.learning_rate = kwargs.get("learning_rate",1e-2)  # 学习率
        self.filter_sizes = kwargs.get("filter_sizes",(2, 3, 4))  # 卷积核尺寸
        self.num_filters = kwargs.get("num_filters",256)  # 卷积核数量(channels数)
        self.MAX_VOCAB_SIZE = kwargs.get("MAX_VOCAB_SIZE",12000)
        self.padding_size = kwargs.get("padding_size",512)  # 每句话处理成的长度(短填长切)
        self.batch_size = kwargs.get("batch_size",4)
        self.RANDOM_SEED = kwargs.get("RANDOM_SEED",2025)


    def get_classlist(self):
        with open("data/datasets/{}/class.txt".format(self.args.dataset),"r",encoding='utf-8') as fl:
            return fl.read().split("\n")

    def __repr__(self):
        return "\n".join(self.class_list)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dims, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_dims)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
        self.mish = Mish()

    def conv_and_pool(self, x, conv):
        x = self.mish(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

