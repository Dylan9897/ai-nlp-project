import os
import sys

import clip
import torch
from loguru import logger
from torch import nn
from torchvision.models import resnet18, resnet50, resnet101


class BaseResnetEncoder(nn.Module):

    def __init__(self, model_path):
        try:
            assert os.path.exists(model_path)
        except:
            logger.error(f'{model_path} does not exist!')
            sys.exit()

        self.model_path = model_path
        self.transform = None
        super(BaseResnetEncoder, self).__init__()

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return feat


class Resnet18Encoder(BaseResnetEncoder):
    def __init__(self, model_path):
        self.name = 'resnet18'

        super(Resnet18Encoder, self).__init__(model_path)
        net = resnet18()
        net.load_state_dict(torch.load(model_path))
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))


class Resnet50Encoder(BaseResnetEncoder):
    def __init__(self, model_path):
        self.name = 'resnet50'

        super(Resnet50Encoder, self).__init__(model_path)
        net = resnet50()
        net.load_state_dict(torch.load(model_path))
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))


class Resnet101Encoder(BaseResnetEncoder):
    def __init__(self, model_path):
        self.name = 'resnet101'

        super(Resnet101Encoder, self).__init__(model_path)
        net = resnet101()
        net.load_state_dict(torch.load(model_path))
        self.backbone = nn.Sequential(*(list(net.children())[:-1]))


class CLIPEncoder(nn.Module):

    def __init__(self, model_path):

        try:
            assert os.path.exists(model_path)
        except:
            logger.error(f'{model_path} does not exist!')
            sys.exit()

        self.model_path = model_path
        self.name = 'vit'

        super(CLIPEncoder, self).__init__()
        self.backbone, self.transform = clip.load(model_path)
        self.backbone = self.backbone.float()

    def forward(self, x):
        feat = self.backbone.encode_image(x)
        return feat

MODELS_DICT = {
    'vit': [CLIPEncoder, 'models/ViT-B-32.pt'],
    'resnet18': [Resnet18Encoder, 'models/resnet18-f37072fd.pth'],
    'resnet50': [Resnet50Encoder, 'models/resnet50-11ad3fa6.pth'],
    'resnet101': [Resnet101Encoder, 'models/resnet101-cd907fc2.pth']
}
