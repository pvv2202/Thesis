import torch
import torch.nn as nn
from utils import broadcastable, get_dim_size

class Instructions:
    '''Instruction Class'''
    def __init__(self):
        '''Initialize Instructions'''
        self.instructions  = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]

    def __call__(self, stack, instruction, *args, **kwargs):
        '''Call instruction on state'''
        # Access the static method from the class, not the instance
        method = getattr(self.__class__, instruction)
        return method(stack, *args, **kwargs)

    #######################
    # Tensor Instructions #
    #######################

    @staticmethod
    def add_previous_dim(stack):
        '''Adds the previous layer's last dimension to the int stack'''
        # TODO: Add the ability to add more than just the last dimension? Could be good for batching to add any dimension
        if len(stack['tensor']) >= 1:
            stack['int'].append(get_dim_size(stack['tensor'][-1], 1))

    '''
    Tensor Stack Operations
    '''
    @staticmethod
    # TODO: Might not want to force input. This doesn't allow for convolution really
    # TODO: Need to rethink modes. The input mode will always be called which is bad. Need to separate them out into 2 bools
    def push_tensor(stack, create=True, input_dim=None, input=False):
        '''Adds a tensor to the stack'''
        # If tensor stack is empty, do nothing
        if len(stack['tensor']) > 0:
            if create and len(stack['int']) >= 1:
                if input and input_dim:
                    # Add tensor to be compatible with input data
                    stack['tensor'].append(torch.randn(input_dim, stack['int'].pop(), requires_grad=True))
                else:
                    # Get second to last dimension
                    dim = get_dim_size(stack['tensor'][-1], 2)
                    stack['tensor'].append(torch.randn(stack['int'].pop(), dim, requires_grad=True))  # Use previous layer to be compatible
                    stack['params'].append(stack['tensor'][-1])
            elif len(stack['params']) >= 1:
                # If we've already created weights, just use them. Should be in order in which they were created in params
                stack['tensor'].append(stack['params'].pop(0))

    @staticmethod
    def duplicate_tensor(stack):
        '''Duplicates the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'][-1]
            stack['tensor'].append(a)

    @staticmethod
    def transpose_tensor(stack):
        '''Transposes the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'].pop()
            result = torch.transpose(a, 0, 1)
            stack['tensor'].append(result)

    '''
    Tensor Operations
    '''
    # TODO: Think about ways to deal with misaligned matrices and how we can smooth any weirdness down the line
    # TODO: For GP, we probably want to have our functions be kind of "smart" i.e. if we specify adding a matmul or something, ensuring that we add a tensor that can handle that? Or just whenever we add a tensor we add one that is compatible with the previous one?
    @staticmethod
    def tensor_matmul(stack):
        '''
        Performs a matrix multiplication on the top two tensors on the stack. Pytorch's matmul function
        automatically supports broadcasting (i.e. multiplying tensors of different sizes for batch/channel dimensions)
        '''
        if len(stack['tensor']) >= 2:
            # Check if sizes are compatible. If not, just return
            if get_dim_size(stack['tensor'][-1], 1) != get_dim_size(stack['tensor'][-2], 2):
                return

            a = stack['tensor'].pop()
            b = stack['tensor'].pop()

            result = torch.matmul(a, b)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_matmul_duplicate(stack):
        '''Performs a matrix multiplication on the top tensor on the stack but duplicates a'''
        if len(stack['tensor']) >= 2:
            # Check if sizes are compatible. If not, just return
            if get_dim_size(stack['tensor'][-1], 1) != get_dim_size(stack['tensor'][-2], 2):
                return

            a = stack['tensor'].pop()
            b = stack['tensor'].pop()

            result = torch.matmul(a, b)
            stack['tensor'].append(result)
            stack['tensor'].append(a)

    @staticmethod
    def tensor_max_pool(stack):
        '''Performs a pooling operation on the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            # If 1-dimensional, do nothing
            if stack['tensor'][-1].dim() == 1:
                return

            # Otherwise, pool
            a = stack['tensor'].pop()

            # Adjust a's shape to be compatible with torch.nn.functional.max_pool2d which expects 4D tensors (batch, channel, height, width)
            if a.dim() == 2:
                a = a.unsqueeze(0).unsqueeze(0)
            elif a.dim() == 3:
                a = a.unsqueeze(0)

            # Pop kernel size and stride from stack
            if len(stack['int']) > 1:
                kernel_size = stack['int'].pop()
                stride = stack['int'].pop() # Don't need constraints on stride. If huge, we will just have 1x1 which should be weeded out via evolution anyway
                if kernel_size < min(a.size(2), a.size(3)): # Only continue if kernel size is compatible (i.e. < height and width of input)
                    result = nn.functional.max_pool2d(a, kernel_size=kernel_size, stride=stride)
                    stack['tensor'].append(result)

    @staticmethod
    def tensor_convolve(stack):
        '''
        Performs a convolution on the top two tensors on the stack Here this is defined as
        multiplying along rows and columns as much as it can.
        '''
        if len(stack['tensor']) >= 2:
            # TODO: Think harder about this. Would there ever be an advantage to projecting a 1D tensor to a 2D tensor and convolving? Maybe include the option and see?
            # If either is 1-dimensional, do nothing
            if stack['tensor'][-1].dim() == 1 or stack['tensor'][-2].dim() == 1:
                return

            a = stack['tensor'].pop()
            b = stack['tensor'].pop() # b will be the kernel

            # Adjust a's shape to be compatible with nn.functional.conv2d which expects 4D tensors (batch, channel, height, width)
            if a.dim() == 2:
                a = a.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, aH, aW]
            elif a.dim() == 3:
                a = a.unsqueeze(0)  # Shape: [1, channels, aH, aW]

            # Adjust b's shape to be compatible with nn.functional.conv2d which expects 4D tensors (out_channels, in_channels, kernel_height, kernel_width)
            # Here, out_channels determines how many channels will be in the output tensor. The rest are dimensions of the kernel
            if b.dim() == 2:
                b = b.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, bH, bW]
                b = b.expand(a.size(1), a.size(1), -1, -1)  # Shape: [in_channels, in_channels, bH, bW]
            elif b.dim() == 3:
                b = b.unsqueeze(1)  # Shape: [C_out, 1, bH, bW]
                b = b.expand(-1, a.size(1), -1, -1)  # Shape: [out_channels, in_channels, bH, bW]

            # TODO: Have 2 separate functions, one that pads and one that doesn't?
            # Calculate padding to achieve 'same' convolution (output size matches input size)
            bH, bW = b.size(2), b.size(3)
            padding = (bH // 2, bW // 2)

            # Perform convolution
            result = nn.functional.conv2d(a, b, padding=padding)

            stack['tensor'].append(result)

    @staticmethod
    def tensor_normalize(stack):
        '''Normalizes the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'].pop()
            result = nn.functional.normalize(a)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_add(stack):
        '''
        Performs an addition on the top two tensors on the stack. Pytorch automatically supports broadcasting
        so we don't need to worry about tensor sizes being the same
        '''
        if len(stack['tensor']) >= 2:
            # If incompatible sizes, do nothing
            if not broadcastable(stack['tensor'][-1], stack['tensor'][-2]):
                return

            a = stack['tensor'].pop()
            b = stack['tensor'].pop()
            result = torch.add(a, b)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_sub(stack):
        '''
        Performs a subtraction on the top two tensors on the stack. Pytorch automatically supports broadcasting
        so we don't need to worry about tensor sizes being the same
        '''
        if len(stack['tensor']) >= 2:
            # If incompatible sizes, do nothing
            if not broadcastable(stack['tensor'][-1], stack['tensor'][-2]):
                return

            a = stack['tensor'].pop()
            b = stack['tensor'].pop()
            result = torch.sub(a, b)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_relu(stack):
        '''Performs a ReLU activation on the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'].pop()
            result = torch.relu(a)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_sigmoid(stack):
        '''Performs a Sigmoid activation on the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'].pop()
            result = torch.sigmoid(a)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_flatten(stack):
        '''Flattens the top tensor on the stack'''
        if len(stack['tensor']) >= 1:
            a = stack['tensor'].pop()
            result = torch.flatten(a)
            stack['tensor'].append(result)

    @staticmethod
    def tensor_divide(stack):
        '''Divides all elements of the top tensor on the stack by a float from the float stack'''
        if len(stack['tensor']) >= 1 and len(stack['float']) >= 1:
            tensor = stack['tensor'].pop()
            divisor = stack['float'].pop()

            result = tensor / divisor
            stack['tensor'].append(result)

    ####################
    # Int Instructions #
    ####################

    @staticmethod
    def int_add(stack):
        '''Adds two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a + b)

    @staticmethod
    def int_mult(stack):
        '''Multiplies two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a * b)

    @staticmethod
    def int_divide(stack):
        '''Divides two integers'''
        if len(stack['int']) >= 2:
            a = stack['int'].pop()
            b = stack['int'].pop()
            stack['int'].append(a // b)

    @staticmethod
    def int_sqrt(stack):
        '''Takes the square root of an integer'''
        if len(stack['int']) >= 1:
            a = stack['int'].pop()
            stack['int'].append(a ** 0.5)

    ######################
    # Float Instructions #
    ######################

    @staticmethod
    def float_add(stack):
        '''Adds two floats'''
        if len(stack['float']) >= 2:
            a = stack['float'].pop()
            b = stack['float'].pop()
            stack['float'].append(a + b)

    @staticmethod
    def float_mult(stack):
        '''Multiplies two floats'''
        if len(stack['float']) >= 2:
            a = stack['float'].pop()
            b = stack['float'].pop()
            stack['float'].append(a * b)

    @staticmethod
    def float_divide(stack):
        # TODO: Might need to be careful here so we don't get enormous values dividing by something
        '''Divides two floats'''
        if len(stack['float']) >= 2:
            # Check so we don't divide by zero
            if stack['float'][-1] != 0:
                a = stack['float'].pop()
                b = stack['float'].pop()
                stack['float'].append(a // b)

    @staticmethod
    def float_sqrt(stack):
        '''Takes the square root of a floats'''
        if len(stack['float']) >= 1:
            a = stack['float'].pop()
            stack['float'].append(a ** 0.5)
