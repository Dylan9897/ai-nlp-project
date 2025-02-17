import torch
import torch.nn as nn
from torchcrf import CRF

class Config():
    """
    相关配置参数
    """
    def __init__(self):
        self.WORD_PAD_ID = None
        self.LABEL_O_ID = None
        self.batch_size = 16
        self.VOCAB_SIZE = 0
        self.EMBEDDING_DIM = 128
        self.HIDDEN_SIZE = 256
        self.TARGET_SIZE = 0
        self.LR = 0.001
        self.EPOCHES = 15
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_step =10
        self.model_path = "ckpt/"


class Model(nn.Module):
    """
    定义BiLSTM+CRF模型
    """
    def __init__(self,config):
        super().__init__()
        self.embed = nn.Embedding(
            config.VOCAB_SIZE,
            config.EMBEDDING_DIM,
            config.WORD_PAD_ID
            )
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

    def _get_lstm_feature(self,input_param):
        out = self.embed(input_param)
        # print(out.shape) # torch.Size([16, 551, 128])
        out,_ = self.lstm(out)
        # print(out.shape) # torch.Size([16, 551, 512])
        return self.linear(out)

    def forward(self,input_param,target,mask):
        y_pred = self._get_lstm_feature(input_param)
        # print(y_pred.shape) # torch.Size([16, 551, 16])
        # print(mask.shape) # torch.Size([16, 551])
        # print(target.shape) # torch.Size([16, 551])
        # s = input()
        return -self.crf.forward(y_pred,target,mask,reduction='mean')

    def predict(self,input_param,mask):
        out = self._get_lstm_feature(input_param)
        return self.crf.decode(out,mask)



