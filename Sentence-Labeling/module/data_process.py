import re
import os
from tqdm import tqdm
from module.utils import read_pkl_file,write_pkl_file

MIN_FREQ = 0
MAX_SIZE = 10000

PAD = "<PAD>"
UNK = "<UNK>"

# 读取文件
def read_file(path):
    content = []
    labels = []
    tc = []
    lc = []
    with open(path,'r',encoding='utf-8') as fl:
        for line in fl.readlines():
            line = line.strip("\ufeff").strip("\n")
            if line == '':
                content.append(tc)
                labels.append(lc)
                tc = []
                lc = []
            else:
                line = re.split("[\s]",line, maxsplit=0, flags=0)
                tc.append(line[0])
                lc.append(line[1])
    return content,labels

# 构造词典
def make_vocab(vocab_path:str,data:list):
    if os.path.exists(vocab_path):
        vocab = read_pkl_file(vocab_path)
    else:
        vocab = {}
        for content in tqdm(data):
            for word in content:
                vocab[word] = vocab.get(word,0)+1
        vocab_list = sorted([_ for _ in vocab.items() if _[1] >= MIN_FREQ],key=lambda x:x[1],reverse=True)[:MAX_SIZE]
        vocab = {word_count[0]:idx for idx,word_count in enumerate(vocab_list)}
        vocab.update({UNK:len(vocab),PAD:len(vocab)+1})
        write_pkl_file(vocab,vocab_path)
    return vocab

