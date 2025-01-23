import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


if __name__ == '__main__':
    from rich import print

    random_logits = torch.rand(8, 2)
    print(random_logits)
    target = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0])
    loss_func = FocalLoss()
    loss = loss_func(random_logits, target)
    print(loss)
    # loss.backward()
