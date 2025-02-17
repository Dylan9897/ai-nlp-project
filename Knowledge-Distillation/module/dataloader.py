import os
import pickle
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset,DataLoader
from module.utils import read_pkl_file,write_pkl_file
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

PAD = "<PAD>"
UNK = "<UNK>"

# 读取文件信息
def read_file(file,analyse=False):
    df = pd.read_csv(file)
    df = shuffle(df)
    if analyse:
        print(f"value counts is {df['class_label'].value_counts()}")
        print(df["class_label"].describe())
    label_id2cate  = dict(enumerate(df.class_label.unique()))
    label_cate2id = {value:key for key,value in label_id2cate.items()}
    df['label'] = df['class_label'].map(label_cate2id)
    return df

# 构造词典
def build_vocab(data,tokenize,max_size,min_freq):
    vocab_dic = {}
    for content in tqdm(data):
        for word in tokenize(content):
            vocab_dic[word] = vocab_dic.get(word,0)+1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],key=lambda x:x[1],reverse=True)[:max_size]
    vocab_dic = {word_count[0]:idx+1 for idx,word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK:len(vocab_dic),PAD:len(vocab_dic)+1})
    return vocab_dic

# 构造数据集
def build_dataset(config,task="st"): 
    """
    task:
        st:student
        tc:teacher
        kd:knowledge-distillation
    """
    df = read_file(config.file_path)
    df_train,df_test = train_test_split(df,test_size=0.1,random_state=config.RANDOM_SEED)
    df_val,df_test = train_test_split(df_test,test_size=0.5,random_state=config.RANDOM_SEED)
    # tokenize = lambda x:config.tokenizer.tokenize(x)


    # print(list(df_train["label"].value_counts()))
    # print(list(df_val["label"].value_counts()))
    # print(list(df_test["label"].value_counts()))
    # print("checkpoint:exam category from datasets")
    # s = input()
    tokenize = lambda x:[y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = read_pkl_file(config.vocab_path)
    else:
        vocab = build_vocab(df_train['content'].values,tokenize,config.MAX_VOCAB_SIZE,min_freq=1)
        write_pkl_file(vocab,config.vocab_path)
    
    def load_dataset(data):
        contents = []
        for i in data.index:
            content = data.loc[i]["content"]
            label = data.loc[i]["label"]
            token = tokenize(content)
            seq_len = len(token)
            if len(token) < config.padding_size:
                token.extend([PAD]*(config.padding_size-len(token)))
            else:
                token = token[:config.padding_size]
                seq_len = config.padding_size
            contents.append((token,int(label),seq_len))
        return contents
    
    train = load_dataset(df_train)
    dev = load_dataset(df_val)
    test = load_dataset(df_test)
    return vocab,train,dev,test
    



def convert2idx(text,vocab):
    vec = [vocab.get(word,vocab[UNK]) for word in text]
    return torch.LongTensor(vec)

class KnowDistillationDataset(Dataset):
    def __init__(self,data,vocab,config):
        self.data = data
        self.vocab = vocab
        self.tokenizer = config.tokenizer
        self.config = config
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        text = self.data[item][0]
        label = self.data[item][1]
        # teacher模型
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config.padding_size,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # student 模型
        embed = convert2idx(text,self.vocab)
        return {
            'texts': ''.join(text),
            'input_ids': encoding['input_ids'].flatten().to(self.config.device),
            'embed':embed.to(self.config.device),
            'attention_mask': encoding['attention_mask'].flatten().to(self.config.device),
            'labels': torch.tensor(label, dtype=torch.long).to(self.config.device)
        }

def create_data_loader(data,vocab,config,batch_size=4):
    ds = KnowDistillationDataset(
        data,
        vocab,
        config
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
    )







