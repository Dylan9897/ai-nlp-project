"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/9/22 15:42
@Email : handong_xu@163.com
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW
import torch.nn.functional as F

BERT_BASE = "ckpt/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(BERT_BASE)

class PaperClassifier(nn.Module):
    def __init__(self, n_classes):
        super(PaperClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_BASE)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        output = self.drop(pooled_output) # dropout
        return self.out(output)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print('Mish activation loaded')

    def forward(self,x):
        x = x*(torch.tanh(F.softplus(x)))
        return x


class Config():
    def __init__(self):
        self.dropout = 0.5
        self.filter_sizes = (2,3,4)
        self.num_filters = 256
        self.embed = 768
        self.num_classes = 7
        self.log_path = "log"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TextCNN_Classifier(nn.Module):
    def __init__(self):
        super(TextCNN_Classifier, self).__init__()
        self.config = Config()
        self.bert = BertModel.from_pretrained(BERT_BASE)
        self.mish = Mish()
        self.dropout = nn.Dropout(self.config.dropout)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1,self.config.num_filters,(k,self.config.embed)) for k in self.config.filter_sizes]
        )
        self.fc = nn.Linear(self.config.num_filters*len(self.config.filter_sizes),self.config.num_classes)

    def conv_and_pool(self,x,conv):
        x = x.unsqueeze(1)
        x = conv(x)
        x = self.mish(x).squeeze(3)
        x = F.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self,input_ids,attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        out = torch.cat([self.conv_and_pool(_,conv) for conv in self.convs],1 )
        out = self.dropout(out)
        out = self.fc(out)
        return out



if __name__ == '__main__':
    from run_TextCNN import train_data_loader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = next(iter(train_data_loader))

    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    model = TextCNN_Classifier()
    model.to(device)
    res = F.softmax(model(input_ids,attention_mask),dim=1)
    print(res)





