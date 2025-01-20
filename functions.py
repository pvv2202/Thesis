import torch
import torch.nn.functional as F

def matmul(x, y):
    return torch.matmul(x, y)

def mat_add(x, y):
    return torch.add(x, y)

def max_pool(x, kernel_size=(2,2), stride=None, padding=0):
    return F.max_pool2d(x, kernel_size, stride, padding)

def flatten(x, start_dim=1):
    return torch.flatten(x, start_dim=start_dim)

def conv2d(x, kernel, bias, stride=1, padding='same', dilation=1):
    return F.conv2d(x, kernel, bias, stride, padding, dilation)

def dup(x):
    return x