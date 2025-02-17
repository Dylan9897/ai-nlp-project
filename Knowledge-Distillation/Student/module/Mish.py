"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/3/4 14:33
@Email : handong_xu@163.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print('Mish activation loaded')

    def forward(self,x):
        x = x*(torch.tanh(F.softplus(x)))
        return x



