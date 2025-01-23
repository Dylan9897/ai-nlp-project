import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

import sys
sys.path.append("/model/handx/work/image-classification")

from models.modelBase import Conv_Block,Linear_Block,Reshape_Block
from conf.config import vgg16 as param
from module.dataloader import train_data,test_data

print(param)

class Model(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.param = param
        self.save_hyperparameters()
        self.model = self.getModel()
        # self.model = AveragedModel(self.model)
        self.args = args

    def getModel(self):
        layers=[]
        for L in self.param:
            if L[0]=="Conv2D":
                layers.append(Conv_Block(L[1],L[2],L[3],L[4],L[5]))
            elif L[0]=="Maxpool":
                layers.append(torch.nn.MaxPool2d(L[1],L[2],L[3]))
            elif L[0]=="Reshape":
                layers.append(Reshape_Block(L[1]))
            elif L[0]=="Conn":
                layers.append(Linear_Block(L[1],L[2]))
            elif L[0]=="LastLinear":
                layers.append(Linear_Block(L[1],L[2],tag=False))
        return torch.nn.Sequential(*layers)

    def forward(self,x):
        return self.model(x)

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

    # def on_train_end(self):
    #     update_bn(self.datamodule.train_dataloader(), self.model, device=self.device)

    


