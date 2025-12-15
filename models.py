import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

def reset_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

        if isinstance(m, nn.Linear):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bound, bound)

def load_backbone(args):
    name = args.model
    backbone = models.__dict__[name.split('_')[-1]](zero_init_residual=True)
    if name.startswith('cifar_'):
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    args.num_backbone_features = backbone.fc.weight.shape[1]
    backbone.fc = nn.Identity()
    reset_parameters(backbone)
    return backbone


def load_mlp(n_in, n_hidden, n_out, num_layers=3, last_bn=True):
    layers = []
    for i in range(num_layers-1):
        layers.append(nn.Linear(n_in, n_hidden, bias=False))
        layers.append(nn.BatchNorm1d(n_hidden))
        layers.append(nn.ReLU())
        n_in = n_hidden
    layers.append(nn.Linear(n_hidden, n_out, bias=not last_bn))
    if last_bn:
        layers.append(nn.BatchNorm1d(n_out))
    mlp = nn.Sequential(*layers)
    reset_parameters(mlp)
    return mlp


class MultiTaskGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks):
        super(MultiTaskGatingNetwork, self).__init__()
        self.gating_fc = nn.ModuleList([
            nn.Linear(input_dim, num_experts) for _ in range(num_tasks)
        ])

    def forward(self, x):
        gating_weights = [F.softmax(gating(x), dim=-1) for gating in self.gating_fc]
        return torch.stack(gating_weights, dim=0)  # Shape: (num_tasks, batch_size, num_experts)