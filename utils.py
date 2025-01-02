import numpy as np
import random

# TODO: Create tests for these functions
# TODO: Fix all of these now that batch isn't included in the shape

def addable(shape1, shape2):
    '''
    Manually checks if two tensors are broadcast-compatible according to broadcasting rules.
    Broadcasting is possible if:
        1. The dimensions are equal, or
        2. One of the dimensions is 1.
    '''
    # Compare from right to left (last dimension to first)
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 != d2 and d1 != 1 and d2 != 1:
            return False
    return True

def add_shape(shape1, shape2):
    '''Returns the shape of the resulting tensor from matrix addition according to PyTorch's rules.'''
    # Convert shapes to lists
    shape1 = list(shape1)
    shape2 = list(shape2)

    # Get the number of dimensions
    dim1 = len(shape1)
    dim2 = len(shape2)

    # Pad the shorter shape with ones on the left (most significant dimensions)
    if dim1 < dim2:
        shape1 = [1] * (dim2 - dim1) + shape1
    elif dim2 < dim1:
        shape2 = [1] * (dim1 - dim2) + shape2

    # Now both shapes have the same length
    result_shape = []
    # Iterate over the dimensions from left to right
    for d1, d2 in zip(shape1, shape2):
        if d1 == d2 or d1 == 1 or d2 == 1:
            # If dimensions are equal or one of them is 1, broadcasting is possible
            result_shape.append(max(d1, d2))
        else:
            # Otherwise, broadcasting is not possible. Should never happen because we check beforehand,
            # but just in case
            return None

    return tuple(result_shape)

def multable(shape1, shape2):
    """Checks if the two shapes are compatible for PyTorch matrix multiplication."""
    result_shape = mult_shape(shape1, shape2)
    return result_shape is not None

def mult_shape(shape1, shape2):
    '''
    Returns the shape of the resulting tensor from a matrix multiplication
    between tensors of shape1 and shape2 according to PyTorch's matmul rules.
    '''

    # Convert shapes to list
    shape1 = list(shape1)
    shape2 = list(shape2)

    # Get the number of dimensions
    dim1 = len(shape1)
    dim2 = len(shape2)

    # Handle 1-D tensors
    if dim1 == 1 and dim2 == 1:
        # Dot product: returns a scalar
        if shape1[0] != shape2[0]:
            return None
        return ()
    elif dim1 == 1 and dim2 >= 2:
        # Prepend 1 to shape1
        shape1 = [1] + shape1
        result_shape = batched_matmul_shape(shape1, shape2)
        if result_shape is None:
            return None
        # Return without prepended 1
        return result_shape[1:]
    elif dim1 >= 2 and dim2 == 1:
        # Append 1 to shape2
        shape2 = shape2 + [1]
        result_shape = batched_matmul_shape(shape1, shape2)
        if result_shape is None:
            return None
        # Return without appended 1
        return result_shape[:-1]
    else:
        # Both tensors are at least 2-D
        result_shape = batched_matmul_shape(shape1, shape2)
        if result_shape is None:
            return None
        # Return the result shape
        return result_shape

def batched_matmul_shape(shape1, shape2):
    """Computes the output shape for batched matrix multiplication"""

    # Ensure both shapes have at least 2 dimensions
    if len(shape1) < 2 or len(shape2) < 2:
        return None

    # Matrix dimensions (excluding batch and channel)
    m1, n1 = shape1[-2], shape1[-1]
    m2, n2 = shape2[-2], shape2[-1]

    # Check if inner dimensions match
    if n1 != m2:
        return None

    # Other dimensions if they exist
    batch_shape1 = shape1[:-2]
    batch_shape2 = shape2[:-2]

    # Broadcast batch dimensions using numpy
    try:
        broadcast_shape = np.broadcast_shapes(batch_shape1, batch_shape2)
    except ValueError as e:
        print(f'Ignore this warning: {e}')
        return None

    # Resulting shape
    result_shape = list(broadcast_shape) + [m1, n2]
    return tuple(result_shape)

def conv2dable(matrix_shape, kernel_shape, stride=1, padding=0, dilation=1):
    '''
    Checks if a 2D convolution operation is possible between a tensor of shape matrix and a kernel of shape kernel.
    Supports asymmetric kernels and matrices. Bias is added by default.
    '''
    shape = conv2d_shape(matrix_shape, kernel_shape, stride, padding, dilation)
    if shape is None or any(dim < 1 for dim in shape):
        return False

    return True

def conv2d_shape(matrix_shape, kernel_shape, stride=1, padding='valid', dilation=1):
    '''
    Returns the shape of the resulting tensor from a 2D convolution operation
    between a tensor of shape matrix and a kernel of shape kernel. Supports
    asymmetric kernels and matrices. Bias is added by default
    '''
    # If the input is under 3D (channel, h, w), or the kernel is not 3D (channel, h, w), we no-op
    if len(matrix_shape) < 3:
        return None

    h_in, w_in = matrix_shape[-2:]
    c_ko, c_ki, h_k, w_k = kernel_shape

    # Valid preserves the same shape
    if padding == 'valid':
        return (c_ko, h_in, w_in)

    # Compute the output dimensions
    h_out = (h_in + 2 * padding - dilation * (h_k - 1) - 1) // stride + 1
    w_out = (w_in + 2 * padding - dilation * (w_k - 1) - 1) // stride + 1

    # Return the output shape (batch/channel are the same)
    return (c_ko, h_out, w_out)

def maxpool2d_shape(matrix_shape, kernel_size, stride=2):
    '''
    Returns the shape of the resulting tensor from a 2D max pooling operation
    '''
    # If the input is under 3D (channel, h, w), or the kernel is not 3D (channel, h, w), we no-op
    if len(matrix_shape) < 3:
        return None

    c, h_in, w_in = matrix_shape[-3:]
    h_k, w_k = kernel_size

    # Default stride is the kernel size
    if stride is None:
        stride = kernel_size

    # Compute the output dimensions
    h_out = (h_in - h_k) // stride + 1
    w_out = (w_in - w_k) // stride + 1

    # Return the output shape (batch/channel are the same)
    return (c, h_out, w_out)

def median_absolute_deviation(data):
    """Calculate the Median Absolute Deviation (MAD)."""
    median = np.median(data)
    deviations = np.abs(data - median)
    return np.median(deviations)


