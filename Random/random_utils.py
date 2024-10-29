import torch

def broadcastable(tensor1, tensor2):
    '''
    Manually checks if two tensors are broadcast-compatible according to broadcasting rules.
    Broadcasting is possible if:
        1. The dimensions are equal, or
        2. One of the dimensions is 1.
    '''
    dims1 = list(tensor1.size())  # List of dimensions for tensor1
    dims2 = list(tensor2.size())  # List of dimensions for tensor2

    # Compare from right to left (last dimension to first)
    for d1, d2 in zip(reversed(dims1), reversed(dims2)):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False
    return True

def mul_broadcastable(shape1, shape2):
    """
    Checks if the batch dimensions of two tensors are broadcastable according to PyTorch's rules.
    """
    len1 = len(shape1) - 2
    len2 = len(shape2) - 2
    for i in range(1, max(len1, len2) + 1):
        s1 = shape1[-i - 2] if i <= len1 else 1
        s2 = shape2[-i - 2] if i <= len2 else 1
        if s1 != s2 and s1 != 1 and s2 != 1:
            return False
    return True


def get_dim_size(tensor, i):
    '''
    Returns the relevant dimension of the tensor on the stack. i corresponds to the dimension we care about in
    reverse order (i.e. if i = 1, and A = (x, y, z, w), we will get w not x. If the tensor is multidimensional:
        If b = (z, b) we only care about z for the next tensor to be compatible so that we have
        a * b = (a, z) * (z, b) = (a, b). If there is a first or second dimension it doesn't matter since
        matmul broadcasts.
    If the tensor is 1D, it just returns the size. PyTorch automatically treats 1D tensors as column or row vectors
    depending on order, so for b = (d) and a = (a, d), a * b = (a, d) * (d, 1) = (a)
    '''
    # If we somehow have a scalar tensor, just return 1
    if tensor.dim() == 0:
        return 1
    # If the tensor is 1 dimensional we just return the size
    elif tensor.dim() == 1:
        return tensor.size()[0]
    # Otherwise return specified index
    else:
        return tensor.size()[-i]