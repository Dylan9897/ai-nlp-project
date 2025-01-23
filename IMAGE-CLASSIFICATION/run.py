import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from importlib import import_module
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import argparse

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

parser = argparse.ArgumentParser(description='Image Classification')
parser.add_argument("--model", type=str, default="AlexNet", help='choose a model:[AlexNet,VggNet,GoogleNet]')
parser.add_argument("--nc",default=1,type=int,help='numbers of GPU')
parser.add_argument("--optim",default=None,help="choose an optimizer func")
parser.add_argument("--pc",default=None,help="choose a precision")
parser.add_argument("--epoch",default=20,help="epoches")
args = parser.parse_args()

# 监视 val_loss
checkpoint_callback = ModelCheckpoint(monitor="val_loss")

trainer = Trainer(
    max_epochs=20,
    gpus = min(args.nc, torch.cuda.device_count()),  # 使用gpu的个数
    devices = [0,1,2,3],                             # gpu可使用的编号
    accelerator="gpu",                               # 设备选择gpu
    callbacks = [checkpoint_callback],
    check_val_every_n_epoch=1,                       # 设置epoch校验的频率
    logger=TensorBoardLogger('lightning_logs/', name='swa_'+args.model),
)

if __name__=="__main__":
    model_name = args.model
    x = import_module("models."+model_name)
    model = x.Model(args)
    trainer.fit(model)
    trainer.test(model)









