import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF
from transformers import BertModel, BertTokenizer, AdamW

BERT_BASE = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(BERT_BASE)

class Config():
    def __init__(self):
        self.batch_size = 16
        self.VOCAB_SIZE = 0
        self.EMBEDDING_DIM = 768
        self.HIDDEN_SIZE = 256
        self.TARGET_SIZE = 0
        self.LR = 0.001
        self.EPOCHES = 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.print_step =10
        self.model_path = "ckpt/"

class Model(nn.Module):
    """
    定义Bert+BiLSTM+CRF模型
    """
    def __init__(self,config):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(BERT_BASE)
        self.lstm = nn.LSTM(
            config.EMBEDDING_DIM,
            config.HIDDEN_SIZE,
            batch_first = True,
            bidirectional=True
        )
        self.linear = nn.Linear(
            2*config.HIDDEN_SIZE,
            config.TARGET_SIZE
        )

        self.crf = CRF(
            config.TARGET_SIZE,
            batch_first=True
        )

        self.drop = nn.Dropout(p=0.3)

    def _get_lstm_feature(self,input_ids,attention_mask):
        #with torch.no_grad():
        output,_ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
            )
        output = self.drop(output)
        out,_ = self.lstm(output)
        return self.linear(out)

    def forward(self,input_ids,attention_mask,target):
        y_pred = self._get_lstm_feature(input_ids,attention_mask)
        return -self.crf.forward(y_pred,target,attention_mask.bool(),reduction='mean')

    def predict(self,input_ids,attention_mask):
        out = self._get_lstm_feature(input_ids,attention_mask)
        return self.crf.decode(out,attention_mask.bool())

