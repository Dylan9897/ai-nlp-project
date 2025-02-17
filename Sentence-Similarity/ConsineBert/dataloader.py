import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader

from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TOKENIZER_PATH = '../bert-base-chinese'
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取文件
def read_file(file):
    res = []
    with open(file,"r",encoding="utf-8") as fl:
        for i,line in enumerate(fl.readlines()):
            line = line.strip("\n").split("\t")
            res.append(line)
    return res

def collate_fn(batch_data,task="train"):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    按照batch中的最大长度数据，对数据进行padding填充
    Args:
        batch_data: batch数据
        task: train,dev,test
    Returns:
    """
    
    input_ids_list_1, attention_mask_list_1, input_ids_list_2, attention_mask_list_2, labels_list = [], [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp_1 = instance["input_ids_1"].to(device)
        attention_mask_temp_1 = instance["mask_1"].to(device)
        input_ids_temp_2 = instance["input_ids_2"].to(device)
        attention_mask_temp_2 = instance["mask_2"].to(device)
        label_temp = instance["label"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list_1.append(torch.tensor(input_ids_temp_1, dtype=torch.long))
        attention_mask_list_1.append(torch.tensor(attention_mask_temp_1, dtype=torch.long))
        input_ids_list_2.append(torch.tensor(input_ids_temp_2, dtype=torch.long))
        attention_mask_list_2.append(torch.tensor(attention_mask_temp_2, dtype=torch.long))
        labels_list.append(label_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids_1": pad_sequence(input_ids_list_1, batch_first=True, padding_value=0),
            "attention_mask_1": pad_sequence(attention_mask_list_1, batch_first=True, padding_value=0),
            "input_ids_2": pad_sequence(input_ids_list_2, batch_first=True, padding_value=0),
            "attention_mask_2": pad_sequence(attention_mask_list_2, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long).to(device)}

class SimDataset(Dataset):
    def __init__(self,file_path,token_path,max_len,task="train"):
        super(SimDataset,self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(token_path)
        self.max_len = max_len
        self.df = read_file(file_path)
        self.data_set = []
        self.get_dataset()

    def encode(self,text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    def get_dataset(self):
        for line in self.df:
            sent1 = line[0]
            sent2 = line[1]
            label = int(line[2])
            tokens_1 = self.encode(sent1)
            tokens_2 = self.encode(sent2)
            self.data_set.append({
                "input_ids_1":tokens_1["input_ids"].flatten(),
                "mask_1":tokens_1["attention_mask"].flatten(),
                "input_ids_2":tokens_2["input_ids"].flatten(),
                "mask_2":tokens_2["attention_mask"].flatten(),
                "label":torch.tensor(label, dtype=torch.long)
            })
        
    def __len__(self):
        return len(self.data_set)

    def __getitem__(self,idx):
        return self.data_set[idx]

train_set=SimDataset("../data/bq_corpus/train.tsv",TOKENIZER_PATH,16)
valid_set=SimDataset("../data/bq_corpus/dev.tsv",TOKENIZER_PATH,16)
traindataloader = DataLoader(train_set,BATCH_SIZE,shuffle=False,collate_fn=collate_fn)
validdataloader = DataLoader(valid_set,BATCH_SIZE,shuffle=False,collate_fn=collate_fn)




