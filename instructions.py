import torch.nn.functional as F
import utils
from dag import *
from utils import *
import inspect

# TODO: Add a bool stack for things like bias in conv

class Instructions:
    '''Instructions for the Push Interpreter'''
    def __init__(self):
        '''Initialize Instructions'''
        self.instructions  = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and not func.startswith("process")]

    def __call__(self, dag, stacks, device, instruction):
        '''Call instruction on state'''
        # Access the static method from the class, not the instance
        method = getattr(self.__class__, instruction)
        # Inspect the method's signature
        sig = inspect.signature(method)
        kwargs = {}
        # Prepare arguments based on the method's parameters
        for param in sig.parameters.values():
            if param.name == 'dag':
                kwargs['dag'] = dag
            elif param.name == 'stacks':
                kwargs['stacks'] = stacks
            elif param.name == 'device':
                kwargs['device'] = device
        return method(**kwargs)

    #########################
    # Matrix Multiplication #
    #########################

    @staticmethod
    def matmul(dag, stacks, device):
        '''Matrix Multiplication'''
        # Do nothing if there are no nodes or integers
        if len(stacks['node']) < 1 or len(stacks['int']) < 1:
            return

        # Pop the top node and integer from the stack
        pop_node = stacks['node'].pop()
        pop_int = stacks['int'].pop()

        # Calculate new dimension
        pop_shape = pop_node.shape
        if len(pop_shape) < 2: # If less than 2, just make them the same (i.e. 1D tensor)
            new_shape = pop_shape
        else: # Otherwise, make the second to last dimension the same as the popped node's last dimension
            new_shape = (pop_shape[-1], pop_int)

        # Create weights
        weights = torch.randn(new_shape, requires_grad=True, device=device)
        stacks['params'].append(weights) # Add weights to the parameters stack

        # Create new node with the output shape of the matrix multiplication
        node = Node(
            shape=utils.mult_shape(pop_shape, new_shape),
            layer=pop_node.layer + 1,
            fn=lambda x, y: torch.matmul(x, y),
            parents=[pop_node],
            weight_id=len(stacks['params']) - 1,
            desc="Matmul"
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        stacks['node'].append(node)

    @staticmethod
    def matmul_dup(dag, stacks, device):
        '''Matrix Multiplication with Duplicate'''
        # Get the top node from the stack
        pop_node = stacks['node'][-1]
        # Call matmul
        Instructions.matmul(dag, stacks, device)
        # Add top node back
        stacks['node'].append(pop_node)

    @staticmethod
    def matmul_stack(dag, stacks):
        '''Matrix Multiplication with Top 2 From Stack'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 2:
            return

        # If not multiplicable, return (noop)
        if not utils.multable(stacks['node'][-2].shape, stacks['node'][-1].shape):
            return

        # Pop the top 2 nodes from the stack
        pop_node1 = stacks['node'].pop()
        pop_node2 = stacks['node'].pop()

        # Create new node
        node = Node (
            shape=utils.mult_shape(pop_node1.shape, pop_node2.shape), # Get the shape of the resulting tensor
            layer=max(pop_node1.layer, pop_node2.layer) + 1, # Take the max layer of the two nodes and add 1
            fn=lambda x, y: torch.matmul(x, y),
            parents=[pop_node1, pop_node2],
            desc="Matmul_Stack"
        )
        dag.add_edge(u=pop_node1, v=node) # Edge between node1 and new node
        dag.add_edge(u=pop_node2, v=node) # Edge between node2 and new node

        stacks['node'].append(node)


    #########################
    ###### Convolution ######
    #########################

    @staticmethod
    def maxpool2d(dag, stacks):
        '''2D Max Pooling'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 1:
            return

        # Check if the top node's shape has 4 dimensions (batch, channel, height, width)
        if len(stacks['node'][-1].shape) < 4:
            return

        res_shape = utils.conv2d_shape(stacks['node'][-1].shape, (stacks['node'][-1].shape[1], stacks['node'][-1].shape[1], 2, 2), stride=2)

        if res_shape[-1] < 1:
            return

        # Pop the top node from the stack
        pop_node = stacks['node'].pop()

        # Create new node
        node = Node(
            shape=res_shape,
            layer=pop_node.layer + 1,
            fn=lambda x: F.max_pool2d(x, kernel_size=2, stride=2), # For now, hardcode kernel size and stride
            parents=[pop_node],
            desc="Maxpool2d"
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        stacks['node'].append(node)

    # TODO: Add a weird convolution that doesn't use conv2d but uses matmul?

    @staticmethod
    def flatten(dag, stacks):
        '''Flatten'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 1:
            return

        # Ensure top node has more than 1 dimension
        if len(stacks['node'][-1].shape) < 2:
            return

        # Pop the top node from the stack
        pop_node = stacks['node'].pop()
        last_shape = pop_node.shape

        prod = 1
        for x in last_shape[1:]:
            prod *= x

        # Create new node
        node = Node(
            shape=(last_shape[0], prod),
            layer=pop_node.layer + 1,
            fn=lambda x: torch.flatten(x, start_dim=1),
            parents=[pop_node],
            desc="Flatten"
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        stacks['node'].append(node)

    # TODO: Add support for asymmetry, dilation, variable stride.
    @staticmethod
    def conv2d(dag, stacks, device):
        '''2D Convolution'''
        # Do nothing if there aren't enough nodes or integers in the stack
        if len(stacks['node']) < 1 or len(stacks['int']) < 2:
            return
        # Check if the top node's shape has 4 dimensions (batch, channel, height, width)
        if len(stacks['node'][-1].shape) < 4:
            return
        # Check if kernel size is valid
        if stacks['int'][-1] > stacks['node'][-1].shape[-1] or stacks['int'][-1] > stacks['node'][-1].shape[-2]:
            return

        # Pop the top node, kernel size, and number of filters from the stack
        pop_node = stacks['node'].pop()
        kernel_size = stacks['int'].pop()
        num_filters = stacks['int'].pop()

        # Define the kernel shape based on the number of input and output channels (filters_
        in_channels = pop_node.shape[1]
        kernel = torch.randn(num_filters, in_channels, kernel_size, kernel_size, requires_grad=True, device=device)  # (out_channels, in_channels, height, width)

        # Bias term for each filter (output channel)
        bias = torch.randn(num_filters, requires_grad=True, device=device)

        # Add the kernel and bias to the 'params' stack
        stacks['params'].extend([kernel, bias])

        node = Node(
            shape=utils.conv2d_shape(pop_node.shape, kernel.shape),
            layer=pop_node.layer + 1,
            fn=lambda x: F.conv2d(input=x, weight=kernel, bias=bias, stride=1, padding=0, dilation=1),
            parents=[pop_node],
            desc="Conv2d"
        )
        dag.add_edge(u=pop_node, v=node)

        stacks['node'].append(node)

    #########################
    #### Matrix Addition ####
    #########################

    @staticmethod
    def mat_add(dag, stacks, device):
        '''Matrix Addition'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 1:
            return

        # Pop the top node from the stack
        pop_node = stacks['node'].pop()

        # Create weights of same shape as popped node
        weights = torch.randn(pop_node.shape, requires_grad=True, device=device)
        stacks['params'].append(weights) # Add weights to the parameters stack

        # Create new node
        node = Node (
            shape=pop_node.shape, # Shape is the same as the popped node
            layer=pop_node.layer + 1, # Take the max layer of the two nodes and add 1
            fn=lambda x, y: torch.add(x, y),
            parents=[pop_node],
            weight_id=len(stacks['params']) - 1,
            desc="Mat_Add"
        )
        dag.add_edge(u=pop_node, v=node) # Edge between node1 and new node

        stacks['node'].append(node)

    @staticmethod
    def mat_add_stack(dag, stacks):
        '''Matrix Addition of Nodes on Stack'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 2:
            return

        # If not addable, return (noop)
        if not utils.addable(stacks['node'][-2].shape, stacks['node'][-1].shape):
            return

        # Pop the top 2 nodes from the stack
        pop_node1 = stacks['node'].pop()
        pop_node2 = stacks['node'].pop()

        # Create new node
        node = Node (
            shape=utils.add_shape(pop_node1.shape, pop_node2.shape), # Get the shape of the resulting tensor
            layer=max(pop_node1.layer, pop_node2.layer) + 1, # Take the max layer of the two nodes and add 1
            fn=lambda x, y: torch.add(x, y),
            parents=[pop_node1, pop_node2],
            desc="Mat_Add_Stack"
        )
        dag.add_edge(u=pop_node1, v=node)

        stacks['node'].append(node)

    @staticmethod
    def mat_add_dup(dag, stacks, device):
        '''Matrix Addition with Duplicate'''
        # Get the top node from the stack
        pop_node = stacks['node'][-1]
        # Call mat_add
        Instructions.mat_add(dag, stacks, device)
        # Add top node back
        stacks['node'].append(pop_node)

    #########################
    ###### PyTorch Ops ######
    #########################

    @staticmethod
    def process_torch_ops(dag, stacks, fn, desc):
        '''Pop the top 2 tensors from the tensor stack'''
        # Do nothing if there aren't enough tensors in the stack
        if len(stacks['node']) < 1:
            return

        # Pop the top node from the stack
        pop_node = stacks['node'].pop()

        # Create new node
        node = Node(
            shape=pop_node.shape,
            layer=pop_node.layer + 1,
            fn=lambda x: fn(x),
            parents=[pop_node],
            desc=desc
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        stacks['node'].append(node)

    # Activation Functions
    @staticmethod
    def relu(dag, stacks):
        '''ReLU Activation Function'''
        Instructions.process_torch_ops(dag, stacks, torch.relu, "ReLU")

    @staticmethod
    def sigmoid(dag, stacks):
        '''Sigmoid Activation Function'''
        Instructions.process_torch_ops(dag, stacks, torch.sigmoid, "Sigmoid")

    # @staticmethod
    # def softmax(dag, stacks):
    #     '''Softmax Activation Function'''
    #     Instructions.process_torch_ops(dag, stacks, lambda x: torch.softmax(x, dim=1), "Softmax")

    # Others
    @staticmethod
    def normalize(dag, stacks):
        '''Normalize'''
        Instructions.process_torch_ops(dag, stacks, torch.nn.functional.normalize, "Normalize")

    #########################
    ### Matrix Scalar Ops ###
    #########################

    @staticmethod
    def process_mat_scalar_ops(dag, stacks, fn, desc):
        '''Pop the top tensor from the tensor stack and the top int from the int stack'''
        # Do nothing if there aren't enough tensors in the stack
        if len(stacks['node']) < 1:
            return

        # Pop the top node and int from the stack
        pop_node = stacks['node'].pop()

        # Create new node
        node = Node(
            shape=pop_node.shape,
            layer=pop_node.layer + 1,
            fn=fn,
            parents=[pop_node],
            desc=desc
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        stacks['node'].append(node)

    # @staticmethod
    # def mat_add_int(dag, stacks):
    #     '''Matrix Addition with Int'''
    #     # Do nothing if there aren't enough ints in the stack
    #     if len(stacks['int']) < 1:
    #         return
    #
    #     pop_int = stacks['int'].pop()
    #     Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.add(x, pop_int), "Mat_Add_Int")

    # @staticmethod
    # def mat_add_float(dag, stacks):
    #     '''Matrix Addition with Float'''
    #     # Do nothing if there aren't enough floats in the stack
    #     if len(stacks['float']) < 1:
    #         return
    #
    #     pop_float = stacks['float'].pop()
    #     Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.add(x, pop_float), "Mat_Add_Float")
    #
    # @staticmethod
    # def mat_mult_float(dag, stacks):
    #     '''Matrix Multiplication with Float'''
    #     # Do nothing if there aren't enough floats in the stack
    #     if len(stacks['float']) < 1:
    #         return
    #
    #     pop_float = stacks['float'].pop()
    #     Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.mul(x, pop_float), "Mat_Mult_Float")

    #########################
    ######## Int Ops ########
    #########################

    @staticmethod
    def process_ints(stacks, fn):
        '''Pop the top 2 ints from the int stack'''
        # Do nothing if there aren't enough integers in the stack
        if len(stacks['int']) < 2:
            return

        fn(stacks['int'].pop(), stacks['int'].pop())

    # @staticmethod
    # def add_int(stacks):
    #     '''Add ints from int stack'''
    #     Instructions.process_ints(stacks, lambda x, y: x + y)

    @staticmethod
    def mult_int(stacks):
        '''Multiply ints from int stack'''
        Instructions.process_ints(stacks, lambda x, y: x * y)

    # @staticmethod
    # def dup_int(stacks):
    #     '''Duplicate the top int on the int stack'''
    #     if len(stacks['int']) < 1:
    #         return
    #     stacks['int'].append(stacks['int'][-1])
    #
    # #########################
    # ####### Float Ops #######
    # #########################
    #
    # @staticmethod
    # def process_floats(stacks, fn):
    #     '''Pop the top 2 floats from the float stack'''
    #     # Do nothing if there aren't enough floats in the stack
    #     if len(stacks['float']) < 2:
    #         return
    #
    #     fn(stacks['float'].pop(), stacks['float'].pop())
    #
    # @staticmethod
    # def add_float(stacks):
    #     '''Add floats from float stack'''
    #     Instructions.process_floats(stacks, lambda x, y: x + y)
    #
    # @staticmethod
    # def sub_float(stacks):
    #     '''Subtract floats from float stack'''
    #     Instructions.process_floats(stacks, lambda x, y: x - y)
    #
    # @staticmethod
    # def mult_float(stacks):
    #     '''Multiply floats from float stack'''
    #     Instructions.process_floats(stacks, lambda x, y: x * y)
    #
    # @staticmethod
    # def div_float(stacks):
    #     '''Divide floats from float stack'''
    #     Instructions.process_floats(stacks, lambda x, y: x / y)
    #
    # @staticmethod
    # def dup_float(stacks):
    #     '''Duplicate the top float on the float stack'''
    #     if len(stacks['float']) < 1:
    #         return
    #     stacks['float'].append(stacks['float'][-1])