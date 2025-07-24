# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from .backbones import resnet18, resnet50
from utils import concat_all_gather

class SSOLEEncoder(nn.Module):

    def __init__(self, base_encoder, dim=128, mlp=False, normalize=True, mlp_layers=2, mlp_hidden_dim=None, mlp_bn=False):

        super(SSOLEEncoder, self).__init__()

        self.encoder = eval(base_encoder)(num_classes=dim)

        self.normalize = normalize

        if mlp:
            x = []
            in_size = self.encoder.fc.weight.shape[1]
            if mlp_hidden_dim is None:
                mlp_hidden_dim = in_size
            for _ in range(mlp_layers - 1):
                if mlp_bn:
                    x.append(nn.Linear(in_size, mlp_hidden_dim, bias=False))
                    x.append(nn.BatchNorm1d(mlp_hidden_dim))
                else:
                    x.append(nn.Linear(in_size, mlp_hidden_dim))
                x.append(nn.ReLU(inplace=True))
                in_size = mlp_hidden_dim
            x.append(nn.Linear(in_size, dim))
            self.encoder.fc = nn.Sequential(*x)

    def forward_backbone(self, x):
        return self.encoder(x)

    def forward_head(self, x):
        if self.normalize:
            x = nn.functional.normalize(x, dim=-1, p=2)
        else:
            x = x.float()
        return x

    def forward(self, inputs, eval_mode=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        start_idx = 0
        for img in inputs:
            if len(img.shape) == 5:
                _x = self.forward_backbone(img.reshape((-1,)+img.shape[2:]))
                _x = _x.reshape(img.shape[:2]+_x.shape[1:])
            else:
                _x = self.forward_backbone(img).unsqueeze(1)
            if start_idx == 0:
                x = _x
            else:
                x = torch.cat((x, _x), dim=1)
            start_idx += 1
        if eval_mode:
            return x.squeeze(1)
        x = self.forward_head(x)
        return x

def ssole_r18(**kwargs):
    return SSOLEEncoder(resnet18, **kwargs)


def ssole_r50(**kwargs):
    return SSOLEEncoder(resnet50, **kwargs)