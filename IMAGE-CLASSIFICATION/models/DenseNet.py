import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

import sys
sys.path.append("/model/handx/work/image-classification")

from models.modelBase import Conv_Block,Linear_Block,Reshape_Block,BC_Block
from conf.config import ResNet as param
from module.dataloader import train_data,test_data
from module.Mish import Mish

class DenseBlock(nn.Module):
    def __init__(self,in_channel,growth_rate,num_layers):
        super().__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(BC_Block(channel,growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self,x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out,x),dim=1)
        return x

# 定义过渡层
def Transition(in_channel,out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel,out_channel,1),
        nn.AvgPool2d(2,2)
    )
    return trans_layer

class Model(pl.LightningModule):
    def __init__(self,args,verbose=False):
        super().__init__()
        self.block_layers = [6,12,24,16]
        self.growth_rate = 32
        self.block1 = nn.Sequential(
            Conv_Block(3,64,(7,7),(2,2),(3,3)),
            nn.MaxPool2d((3,3),(2,2),(1,1))
        )

        channels = 64
        block = []
        for i,layers in enumerate(self.block_layers):
            block.append(DenseBlock(channels,self.growth_rate,layers))
            channels += layers*self.growth_rate
            if i!= len(self.block_layers)-1:
                block.append(Transition(channels,channels//2))
                channels = channels//2
        
        self.block2 = nn.Sequential(*block)
        self.block2.add_module("bn",nn.BatchNorm2d(channels))
        self.block2.add_module("relu",nn.ReLU())
        self.block2.add_module("avg_pool",nn.AvgPool2d(3))

        self.classifier = nn.Linear(channels,10)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.shape[0],-1)
        x = self.classifier(x)
        return x

    def train_dataloader(self):
        return train_data
    
    def val_dataloader(self):
        return test_data

    def test_dataloader(self):
        return test_data

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

   

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y,task='multiclass', num_classes=10)
        self.log('val acc',acc,prog_bar=True)
        return {"pred":preds}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, y,task='multiclass', num_classes=10)
        self.log('test_loss', loss,prog_bar=True)
        self.log('test acc',acc,prog_bar=True)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__=="__main__":
    # 验证dense block
    # testBlock = DenseBlock(3,12,3)
    # input_demo = Variable(torch.zeros(1,3,96,96))
    # print(testBlock)
    # out = testBlock(input_demo)
    # print(out.shape)
    # 验证过度层
    # testNet = Transition(3,12)
    # input_demo = Variable(torch.zeros(1,3,96,96))
    # print(testNet)
    # out = testNet(input_demo)
    # print(out.shape)
    # 验证DenseNet
    testNet = Model(args=0)
    print(testNet)
    input_demo = Variable(torch.zeros(1,3,96,96))
    out = testNet(input_demo)
    print(out.shape)


