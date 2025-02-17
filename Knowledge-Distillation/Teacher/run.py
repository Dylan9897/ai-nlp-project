import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from models.bert import TextCNN_Classifier,Config,tokenizer
from sklearn.metrics import confusion_matrix, classification_report
from module.dataloader import build_dataset,create_data_loader
from module.train import train
from transformers import AutoConfig,AutoModel,AutoTokenizer,AdamW,get_linear_schedule_with_warmup,logging



parser = argparse.ArgumentParser(description="Long Text Classification")
parser.add_argument('--optim',default=False,help="choose differential learning rate or not")
parser.add_argument('--epoch',default=5,help='num of epoches')
parser.add_argument('--batch',default=4,help='num of batchsize')
args = parser.parse_args()

MAX_LEN = 256
BATCH_SIZE = args.batch
RANDOM_SEED = 2022

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = Config()
config.RANDOM_SEED = RANDOM_SEED
config.file_path = "data/labeled_data.csv"
config.vocab_path = "Student/data/vocab.pkl"
config.padding_size =256
config.MODEL_DIR = "Teacher/saved_dict"
config.optim = args.optim
config.num_epoches = args.epoch


# (df,tokenizer,max_len,batch_size)
vocab,df_train,df_val,df_test = build_dataset(config)

train_data_loader = create_data_loader(df_train,tokenizer,MAX_LEN,BATCH_SIZE)
valid_data_loader = create_data_loader(df_val,tokenizer,MAX_LEN,BATCH_SIZE)
test_data_loader = create_data_loader(df_test,tokenizer,MAX_LEN,BATCH_SIZE)
config.train_examples = len(df_train)
config.valid_examples = len(df_val)
config.test_examples = len(df_test)
model = TextCNN_Classifier()
model = model.to(device)
train(model,train_data_loader,valid_data_loader,test_data_loader,config)