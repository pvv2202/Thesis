import torch
import torch.nn.functional as F

'''
Different functions used. Have to specify these to make instructions pickleable (lambda isn't)
'''

def matmul(x, y):
    return torch.matmul(x, y)

def mat_add(x, y):
    return torch.add(x, y)

def max_pool(x, kernel_size=(2,2), stride=None, padding=0):
    return F.max_pool2d(x, kernel_size, stride, padding)

def avg_pool(x, kernel_size=(2,2), stride=None, padding=0):
    return F.avg_pool2d(x, kernel_size, stride, padding)

def flatten(x, start_dim=1):
    return torch.flatten(x, start_dim=start_dim)

def conv2d(x, kernel, bias, stride=1, padding='same', dilation=1):
    return F.conv2d(x, kernel, bias, stride, padding, dilation)

def dup(x):
    return x

def embedding(x, weights):
    if x.dtype != torch.long:
        x = x.long()

    original_shape = x.shape
    x_flat = x.view(-1)
    embs_flat = weights[x_flat]

    if len(original_shape) == 1:
        return embs_flat
    else:
        batch_size, seq_len = original_shape
        embed_dim = weights.shape[1]
        return embs_flat.view(batch_size, seq_len, embed_dim)

# def batch_norm(x, gamma, beta, eps=1e-5, momentum=0.1, training=True):
#     if training:


