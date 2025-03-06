import torch
import torch.nn as nn
import torch.nn.functional as F

class MatmulNodes(nn.Module):
    def forward(self, x, y):
        return torch.matmul(x, y)

class MatAddNodes(nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

class MatAddWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.weight
