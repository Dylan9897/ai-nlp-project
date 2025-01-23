import pickle
import jieba
import pandas as pd
from logger import logger
from sklearn.feature_extraction.text import TfidfVectorizer


class DataProcessor():
    """
    读取文件
    """
    def __init__(self,args):
        self.cut = args.cws
        
    def _read_file(self,path):
        df = pd.read_excel(path)
        return df
    
    def _return_label_dic(self,data):
        labels = list(data["label"].unique())
        dic = dict(zip(labels,range(len(labels))))
        return dic
    
    def _cws(self,seq):
        return " ".join(jieba.cut(seq))

    def _return_dataset(self,data,label_dic):
        contents = []
        labels = []
        
        for i,elem in enumerate(data["content"]):
            if self.cut:
                contents.append(self._cws(elem))
            else:
                contents.append(" ".join(list(elem)))
            if i< 3:
                logger.info(f"第 {i} 条样本内容：{contents[-1]}")
        for elem in data["label"]:
            labels.append(label_dic[elem])
        return contents,labels
        

def dump(target,object):
    with open(target,'wb') as fw:
        pickle.dump(object,fw)

def undump(object):
    with open(object,'rb') as fw:
        obj = pickle.load(fw)
    return obj


class Converter():
    """
    本代码的功能：
        1、训练tf-idf模型
        2、将句子转换成tf-idf向量
    """
    def __init__(self):
        pass

    def operate(self,seq):
        while '  ' in seq:
            seq = seq.replace('  ',' ')
        return seq

    def tfvectorize(self,words,test=None):
        words = [self.operate(elem) for elem in words]
        v = TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')
        if test == None:
            data = v.fit_transform(words)
            dump('ckpt/tfidf.model',v)
        else:
            v = undump('ckpt/tfidf.model')
            data = v.transform(words)
        return data



