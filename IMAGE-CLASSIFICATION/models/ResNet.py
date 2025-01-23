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

from models.modelBase import Conv_Block,Linear_Block,Reshape_Block
from conf.config import ResNet as param
from module.dataloader import train_data,test_data
from module.Mish import Mish

class RasidualBlock(nn.Module):
    def __init__(self,in_channel,out_channel,same_shape=True):
        super().__init__()
        self.same_shape = same_shape
        self.act = Mish()
        stride = 1 if self.same_shape else 2
        self.stride = stride
        self.conv1 = Conv_Block(in_channel,out_channel,(3,3),(stride,stride),(1,1))
        self.conv2 = Conv_Block(out_channel,out_channel,(3,3),(1,1),(1,1))
        if not self.same_shape:
            self.conv3 = Conv_Block(in_channel,out_channel,(1,1),(stride,stride),(0,0))

    def forward(self,x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = self.act(out)
        if not self.same_shape:
            x = self.conv3(x)
        return self.act(x+out)

class Model(pl.LightningModule):
    def __init__(self,args,verbose=False):
        super().__init__()
        self.param = param
        self.save_hyperparameters()
        self.args = args
        self.verbose = verbose

        self.block1 = Conv_Block(3,64,(7,7),(2,2),(0,0))

        self.block2 = nn.Sequential(
            nn.MaxPool2d((3,3),(2,2),(0,0)),
            RasidualBlock(64,64),
            RasidualBlock(64,64)
        )

        self.block3 = nn.Sequential(
            RasidualBlock(64,128,False),
            RasidualBlock(128,128)
        )

        self.block4 = nn.Sequential(
            RasidualBlock(128,256,False),
            RasidualBlock(256,256)
        )

        self.block5 = nn.Sequential(
            RasidualBlock(256,512,False),
            RasidualBlock(512,512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512,10)

    def forward(self,x):
        x=self.block1(x)
        if self.verbose:
            print('block	1	output:	{}'.format(x.shape))
        x=self.block2(x)
        if self.verbose:
            print('block	2	output:	{}'.format(x.shape))
        x=self.block3(x)
        if self.verbose:
            print('block	3	output:	{}'.format(x.shape))
        x=self.block4(x)
        if self.verbose:
            print('block	4	output:	{}'.format(x.shape))
        x=self.block5(x)
        if self.verbose:
            print('block	5	output:	{}'.format(x.shape))
        x=x.view(x.shape[0],-1)
        x=self.classifier(x)
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
    # # test block
    # testBlock = RasidualBlock(3,128,False)
    # x = Variable(torch.zeros(1,3,32,32))
    # out = testBlock(x)
    # print(out.shape)
    # # test ResNet
    x = Variable(torch.zeros(1,3,32,32))
    net = Model(args=0,verbose=True)
    out = net(x)
    print(out.shape)

