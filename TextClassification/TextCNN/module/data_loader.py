import os
import json
from tqdm import tqdm
import jieba
import torch
from torch.utils.data import Dataset,DataLoader
from TextCNN.utils.utils import read_pkl_file,write_pkl_file
import re



PAD = "<PAD>"
UNK = "<UNK>"

def cleaner(seq):
    return re.sub("\s+","",seq)

def convert2idx(text,vocab):
    vec = [vocab.get(word,vocab[UNK]) for word in text]
    return torch.LongTensor(vec)

def read_file(file):
    if not os.path.exists(file):
        return []
    result = []
    with open(file,"r",encoding="utf-8") as fl:
        for line in fl.readlines():
            line = line.strip("\n")
            line = json.loads(line)
            result.append(line)
        return result

def build_vocab(data,tokenize,max_size,min_freq):
    vocab_dic = {}
    for elem in tqdm(data):
        content = elem["content"]
        for word in tokenize(content):
            vocab_dic[word] = vocab_dic.get(word,0)+1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],key=lambda x:x[1],reverse=True)[:max_size]
    vocab_dic = {word_count[0]:idx+1 for idx,word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK:len(vocab_dic),PAD:len(vocab_dic)+1})
    return vocab_dic


def build_dataset(config):
    """
    task:
        st:student
        tc:teacher
        kd:knowledge-distillation
    """
    train_data = read_file(config.train_path)
    dev_data = read_file(config.dev_path)
    test_data = read_file(config.test_path)
    tokenize = lambda x: [y for y in x] if not config.cws else list(jieba.cut(x))
    if os.path.exists(config.vocab_path):
        vocab = read_pkl_file(config.vocab_path)
    else:

        vocab = build_vocab(train_data, tokenize, config.MAX_VOCAB_SIZE, min_freq=1)
        write_pkl_file(vocab, config.vocab_path)

    def load_dataset(data):
        contents = []
        for i in range(len(data)):
            content = cleaner(data[i]["content"])
            label = data[i]["label"]
            token = tokenize(content)
            seq_len = len(token)
            if len(token) < config.padding_size:
                token.extend([PAD] * (config.padding_size - len(token)))
            else:
                token = token[:config.padding_size]
                seq_len = config.padding_size
            contents.append((token, int(label), seq_len))
        return contents

    train = load_dataset(train_data)
    dev = load_dataset(dev_data)
    test = load_dataset(test_data)
    return vocab, train, dev, test


class StudentDataset(Dataset):
    def __init__(self ,data ,vocab ,config):
        self.data = data
        self.vocab = vocab
        self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self ,item):
        text = self.data[item][0]
        label = self.data[item][1]
        embed = convert2idx(text ,self.vocab)
        return {
            'texts': ''.join(text),
            'embed' :embed.to(self.config.device),
            'labels': torch.tensor(label, dtype=torch.long).to(self.config.device)
        }


def create_data_loader(data,vocab,config,batch_size=4):
    ds = StudentDataset(
        data,
        vocab,
        config
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
    )

