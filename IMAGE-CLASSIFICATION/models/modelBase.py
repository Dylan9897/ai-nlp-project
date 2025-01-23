import torch
# import torchsummary
from module.Mish import Mish

class Conv_Block(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Conv_Block,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        # self.act = torch.nn.ReLU()
        self.act = Mish()
    
    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out
    
class Linear_Block(torch.nn.Module):
    def __init__(self,infeatures,outfeatures,tag=True):
        super(Linear_Block,self).__init__()
        self.linear = torch.nn.Linear(infeatures, outfeatures)
        self.tag = tag
        if self.tag:
            # self.act = torch.nn.ReLU()
            self.act = Mish()
    
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        out = self.linear(x)
        if self.tag:
            out = self.act(out)
        return out

class Reshape_Block(torch.nn.Module):
    def __init__(self,shape):
        super(Reshape_Block,self).__init__()
        self.shape = shape
    
    def forward(self,x):
        out = x.reshape(self.shape)
        return out