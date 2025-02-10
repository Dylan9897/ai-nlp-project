import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils import (
    logging,
)

from transformers import Trainer

logger = logging.get_logger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算softmax后的概率
        probs = F.softmax(inputs, dim=1)
        # 获取每个样本对应真实标签的概率
        pt = probs[range(probs.size(0)), targets]

        # 计算log(pt)
        log_pt = torch.log(pt)

        # 如果提供了alpha，则应用类别权重
        if self.alpha is not None:
            at = self.alpha[targets]
            log_pt = log_pt * at

        # 根据Focal Loss公式计算损失
        loss = -((1 - pt) ** self.gamma) * log_pt

        # 根据reduction参数决定返回的损失形式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MyTrainer(Trainer):
    def __init__(self, model, args=None, **kwargs):
        super().__init__(
            model=model, args=args, **kwargs
        )
        self.base_model = model
        self.loss_func = FocalLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs["labels"]
        # print(labels)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss