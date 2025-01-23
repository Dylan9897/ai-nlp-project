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
from conf.config import vgg16 as param
from module.dataloader import train_data,test_data

class Block(nn.Module):
    def __init__(self,in_channels,path1_1,path2_1,path2_3,path3_1,path3_5,path4_1,stride=[1,1],padding=[0,0]):
        super().__init__()
        self.branch1 = Conv_Block(in_channels,path1_1,[1,1],stride,padding)

        self.branch2 = nn.Sequential(
            Conv_Block(in_channels,path2_1,[3,3],stride,[1,1]),
            Conv_Block(path2_1,path2_3,[3,3],stride,[1,1])
        )

        self.branch3 = nn.Sequential(
            Conv_Block(in_channels,path3_1,[1,1],stride,padding),
            Conv_Block(path3_1,path3_5,[5,5],stride,[2,2])
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d([3,3],stride,[1,1]),
            Conv_Block(in_channels,path4_1,[1,1],stride,padding)
        )

    def forward(self,x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1,f2,f3,f4),dim=1)
        return output




class Model(pl.LightningModule):
    def __init__(self,args,verbose=False):
        super().__init__()
        self.param = param
        self.save_hyperparameters()
        self.args = args
        self.verbose = verbose
        

        self.block1 = nn.Sequential(
            Conv_Block(3,64,[7,7],[2,2],[3,3]),
            nn.MaxPool2d(3,2)
        )

        self.block2 = nn.Sequential(
            Conv_Block(64,64,[1,1],[1,1],[0,0]),
            Conv_Block(64,192,[3,3],[1,1],[1,1])
        )

        self.block3 = nn.Sequential(
            Block(192,64,96,128,16,32,32),
            Block(256,128,128,192,32,96,64),
            nn.MaxPool2d(3,2)
        )

        self.block4 = nn.Sequential(
            Block(480,192,96,208,16,48,64),
            Block(512,160,112,224,24,64,64),
            Block(512,128,128,256,24,64,64),
            Block(512,112,144,288,32,64,64),
            Block(528,256,160,320,32,128,128),
            nn.MaxPool2d([3,3],[2,2],[0,0])
        )

        self.block5 = nn.Sequential(
            Block(832,256,160,320,32,128,128),
            Block(832,384,182,384,48,128,128),
            nn.AvgPool2d(1)
        )

        self.classifier = nn.Linear(1024,10)

    def forward(self,x):
        x = self.block1(x)
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
        x=x.view(x.shape[0],	-1)
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
        return torch.optim.Adam(self.parameters(), lr=1e-4)



if __name__ == "__main__":
    # test Block
    # input_demo = Variable(torch.zeros(1,3,64,64))
    # testBlock = Block(3,64,48,64,64,96,32)
    # print(testBlock)
    # out = testBlock(input_demo)
    # print(out.shape)
    # test GoogleNet
    input_demo = Variable(torch.zeros(1,3,64,64))
    testNet = Model(args=0)
    print(testNet)
    out = testNet(input_demo)
    print(out.shape)
