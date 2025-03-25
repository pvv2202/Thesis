import math
import torch.nn.functional as F
from functools import partial
import utils
from dag import *
from functions import *
import inspect
import torch.nn.init as init
import torch.nn as nn
from custom_modules import *

# TODO: Add a bool stack for things like bias in conv

ACTIVATIONS = ['relu', 'sigmoid', 'softmax', 'tanh']

class Instructions:
    """Instructions for the Push Interpreter. Returns True if instruction was successful (added to dag),
    False otherwise"""

    def __init__(self, activation='relu'):
        """Initialize Instructions. If activation is None, all instructions are available. Otherwise, we exclude
        activation functions"""
        if activation is not None:
            self.instructions = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and not func.startswith("process") and func not in ACTIVATIONS]
        else:
            self.instructions = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and not func.startswith("process")]

    def __call__(self, dag, net, stacks, device, instruction, separate_ints):
        """Call instruction on state"""
        # Access the static method from the class, not the instance
        method = getattr(self.__class__, instruction)
        # Inspect the method's signature
        sig = inspect.signature(method)
        kwargs = {}
        # Prepare arguments based on the method's parameters
        for param in sig.parameters.values():
            if param.name == 'dag':
                kwargs['dag'] = dag
            elif param.name == 'net':
                kwargs['net'] = net
            elif param.name == 'stacks':
                kwargs['stacks'] = stacks
            elif param.name == 'device':
                kwargs['device'] = device
            elif param.name == 'separate_ints':
                kwargs['separate_ints'] = separate_ints
        return method(**kwargs)

    #########################
    # Matrix Multiplication #
    #########################

    @staticmethod
    def matmul(dag, net, stacks, device):
        """Matrix Multiplication"""
        # Do nothing if there are no nodes or integers
        if len(net['nodes']) < 1 or len(stacks['int']) < 1 or len(net['nodes'][0].shape) < 1:
            return False

        # Pop the top node from queue, top integer from the stack
        pop_node = net['nodes'].popleft()
        pop_int = stacks['int'].pop()

        # Calculate new dimension
        pop_shape = pop_node.shape
        weight_shape = (pop_shape[-1], pop_int)

        # Calculate flops depending on dimension
        if len(pop_shape) < 2:
            flops = (2 * pop_shape[-1] - 1) * pop_int
        else:
            flops = (2 * pop_shape[-1] * pop_shape[-2] - 1) * pop_int

        matmul_layer = nn.Linear(pop_shape[-1], pop_int, bias=False).to(device)

        # Create new node with the output shape of the matrix multiplication
        node = Node(
            shape=utils.mult_shape(pop_shape, weight_shape),
            layer=pop_node.layer + 1,
            fn=matmul_layer,
            desc="Matmul",
            flops=flops
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node) # Add new node to stack

        return True

    @staticmethod
    def matmul_nodes(dag, net):
        """Matrix Multiplication with Top 2 From Stack"""
        # Do nothing if there aren't enough nodes in the stack, or they're 1D
        if len(net['nodes']) < 2 or len(net['nodes'][0].shape) < 2 or len(net['nodes'][1].shape) < 2:
            return False

        # If not multable, return (noop)
        if not utils.multable(net['nodes'][0].shape, net['nodes'][1].shape):
            return False

        # Pop the top 2 nodes from the stack
        pop_node1 = net['nodes'].popleft()
        pop_node2 = net['nodes'].popleft()

        # Calculate flops depending on dimension
        if len(pop_node1.shape) < 2:
            flops = (2 * pop_node1.shape[-1] - 1) * pop_node2.shape[-1]
        else:
            flops = (2 * pop_node1.shape[-1] * pop_node1.shape[-2] - 1) * pop_node2.shape[-1]

        matmul_nodes_layer = MatmulNodes()

        # Create new node
        node = Node(
            shape=utils.mult_shape(pop_node1.shape, pop_node2.shape),  # Get the shape of the resulting tensor
            layer=max(pop_node1.layer, pop_node2.layer) + 1,  # Take the max layer of the two nodes and add 1
            fn=matmul_nodes_layer,
            desc="Matmul_Nodes",
            flops=flops
        )

        # Add whichever node is lower in the graph so that both will have been processed.
        dag.add_edge(u=pop_node1, v=node)
        dag.add_edge(u=pop_node2, v=node)

        net['nodes'].append(node)

        return True

    #########################
    ###### Convolution ######
    #########################
    # TODO: Flops for these will be dependent on stride and padding if I make that variable as well

    @staticmethod
    def maxpool2d(dag, net):
        """2D Max Pooling"""
        # Do nothing if there aren't enough nodes in the stack
        if len(net['nodes']) < 1:
            return False

        # Check if the top node's shape has 4 dimensions (batch, channel, height, width)
        if len(net['nodes'][0].shape) < 3:
            return False

        # Check if max pooling is possible.
        if not utils.poolable(net['nodes'][0].shape, (2, 2), stride=2):
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()
        new_shape = utils.pool2d_shape(pop_node.shape, (2, 2), stride=2)
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # For now, hardcode kernel size and stride

        # Create new node
        node = Node(
            shape=new_shape,
            layer=pop_node.layer + 1,
            fn=max_pool_layer,  # For now, hardcode kernel size and stride
            desc="Maxpool2d",
            flops=math.prod(new_shape) * (2 * 2 - 1)
            # Comparisons are counted as 1. Make k * k - 1 comparisons for each output element. This is output
            # elements * comparisons/element
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node) # Add new node to stack

        return True

    @staticmethod
    def avgpool2d(dag, net):
        """2D Average Pooling"""
        # Do nothing if there aren't enough nodes in the stack
        if len(net['nodes']) < 1:
            return False

        # Check if the top node's shape has 4 dimensions (batch, channel, height, width)
        if len(net['nodes'][0].shape) < 3:
            return False

        # Check if max pooling is possible.
        if not utils.poolable(net['nodes'][0].shape, (2, 2), stride=2):
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()
        new_shape = utils.pool2d_shape(pop_node.shape, (2, 2), stride=2)
        avg_pool_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=2) # For now, hardcode kernel size and stride

        # Create new node
        node = Node(
            shape=new_shape,
            layer=pop_node.layer + 1,
            fn=avg_pool_layer,
            desc="Avgpool2d",
            flops=math.prod(new_shape) * (2 * 2 - 1)
            # Comparisons are counted as 1. Make k * k - 1 comparisons for each output element. This is output elements * comparisons/element
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node) # Add new node to stack

        return True

    # TODO: Add a weird convolution that doesn't use conv2d but uses matmul?

    @staticmethod
    def flatten(dag, net):
        """Flatten"""
        # Do nothing if there aren't enough nodes in the stack
        if len(net['nodes']) < 1:
            return False

        # Ensure top node has more than 1 dimension
        if len(net['nodes'][0].shape) < 2:
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()
        last_shape = pop_node.shape

        prod = 1
        for x in last_shape:
            prod *= x

        # Define partial function
        flatten_layer = nn.Flatten(start_dim=1)

        # Create new node
        node = Node(
            shape=(prod,),
            layer=pop_node.layer + 1,
            fn=flatten_layer,
            desc="Flatten",
            flops=0
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node) # Add new node to stack

        return True

    # TODO: Add dilation, variable stride?
    @staticmethod
    def conv2d(dag, net, stacks, device, separate_ints):
        """2D Convolution. Just uses PyTorch's conv2d. A bunch of code here but most is just checking for no-op"""
        # First check we have enough nodes and integers
        if separate_ints:
            # Do nothing if there aren't enough nodes or integers (small and normal) in the stack
            if len(net['nodes']) < 1 or len(stacks['sint']) < 1 or len(stacks['int']) < 1:
                return False
        else:
            # Same as above but with only one stack for ints. Sint stack still exists for simplicity but is never used
            if len(net['nodes']) < 1 or len(stacks['int']) < 2:
                return False

        # Check if the top node's shape has 3 dimensions (channel, height, width). Same for both cases
        if net['nodes'][0].shape is None:
            print(net['nodes'])
        if len(net['nodes'][0].shape) < 3:
            return False

        stride = 1 # Default stride

        # Check for valid kernel and whether we can convolve
        if separate_ints:
            # Check if kernel size is valid. Can't be greater than either dimension along the input
            if stacks['sint'][-1] > net['nodes'][0].shape[-1] or stacks['sint'][-1] > net['nodes'][0].shape[-2]:
                return False
            # If we can't convolve, just return. Kernel will be int (out channels), node channels (int channels)
            # sint, sint (kernel size = h, w)
            if not utils.conv2dable(net['nodes'][0].shape, (
                    stacks['int'][-1], net['nodes'][0].shape[1], stacks['sint'][-1], stacks['sint'][-1])):
                return False
        else:
            # Check if kernel size is valid. Can't be greater than either dimension along the input
            if stacks['int'][-1] > net['nodes'][0].shape[-1] or stacks['int'][-1] > net['nodes'][0].shape[-2]:
                return False
            # If we can't convolve, just return
            if not utils.conv2dable(net['nodes'][0].shape, (stacks['int'][-2], net['nodes'][0].shape[1], stacks['int'][-1], stacks['int'][-1])):
                return False

        # Pop the top node, kernel size, and number of filters from the stack
        pop_node = net['nodes'].popleft()

        if separate_ints:
            kernel_size = stacks['sint'].pop()
        else:
            kernel_size = stacks['int'].pop()

        num_filters = stacks['int'].pop()

        padding = 'same'

        # Check what stride will be and assign padding accordingly. Can't do same with stride > 1
        if separate_ints:
            if len(stacks['sint']) > 0 and stacks['sint'][-1] > 1:
                padding = 0
        else:
            if len(stacks['int']) > 0 and stacks['int'][-1] > 1:
                padding = 0

        # Check if there is a valid stride in the int stack. If so, use that.
        if separate_ints:
            if len(stacks['sint']) > 0 and utils.conv2dable(pop_node.shape, (num_filters, pop_node.shape[1], kernel_size, kernel_size), stacks['sint'][-1], padding):
                stride = stacks['sint'].pop()
            else:
                padding = 'same' # Reset padding if this doesn't work
        else:
            if len(stacks['int']) > 0 and utils.conv2dable(pop_node.shape, (num_filters, pop_node.shape[1], kernel_size, kernel_size), stacks['int'][-1], padding):
                stride = stacks['int'].pop()
            else:
                padding = 'same'  # Reset padding if this doesn't work

        # Define the kernel shape based on the number of input and output channels
        in_channels = pop_node.shape[0]

        conv_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        ).to(device)

        new_shape = utils.conv2d_shape(pop_node.shape, (num_filters, in_channels, kernel_size, kernel_size), padding=padding, stride=stride)

        # TODO: Add different padding options?
        node = Node(
            shape=new_shape,
            layer=pop_node.layer + 1,
            fn=conv_layer,
            desc="Conv2d",
            flops=(in_channels * (kernel_size ** 2) * 2 - 1) * math.prod(new_shape)
            # k * k multiplications, k * k - 1 additions for each output element.
        )
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node)

        return True

    #########################
    #### Matrix Addition ####
    #########################

    @staticmethod
    def mat_add(dag, net, device):
        """Matrix Addition"""
        # Do nothing if there aren't enough nodes in the stack
        if len(net['nodes']) < 1:
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()

        mat_add_layer = MatAddWeight(pop_node.shape).to(device)

        # Create new node
        node = Node(
            shape=pop_node.shape,  # Shape is the same as the popped node
            layer=pop_node.layer + 1,  # Take the max layer of the two nodes and add 1
            fn=mat_add_layer,
            desc="Mat_Add",
            flops=math.prod(pop_node.shape)
        )
        dag.add_edge(u=pop_node, v=node)  # Edge between popped node and new node

        net['nodes'].append(node)

        return True

    @staticmethod
    def mat_add_nodes(dag, net):
        """Matrix Addition of Nodes on Stack"""
        # Do nothing if there aren't enough nodes in the stack
        if len(net['nodes']) < 2:
            return False

        # If not addable, return (noop)
        if not utils.addable(net['nodes'][0].shape, net['nodes'][1].shape):
            return False

        # Pop the top 2 nodes from the stack
        pop_node1 = net['nodes'].popleft()
        pop_node2 = net['nodes'].popleft()

        mat_add_nodes_layer = MatAddNodes()

        # Create new node
        node = Node(
            shape=utils.add_shape(pop_node1.shape, pop_node2.shape),  # Get the shape of the resulting tensor
            layer=max(pop_node1.layer, pop_node2.layer) + 1,  # Take the max layer of the two nodes and add 1
            fn=mat_add_nodes_layer,
            desc="Mat_Add_Nodes",
            flops=math.prod(pop_node1.shape)
        )

        dag.add_edge(u=pop_node1, v=node)
        dag.add_edge(u=pop_node2, v=node)

        net['nodes'].append(node)

        return True

    #########################
    ######## RNN Ops ########
    #########################

    @staticmethod
    def await_connection(dag, net):
        """Create node waiting for a connection"""
        if len(net['nodes']) < 1:
            return False

        id_layer = nn.Identity()

        ref = net['nodes'][0]
        node = Node(
            shape=ref.shape,
            layer=ref.layer,
            fn=id_layer,
            desc="Await Connection",
            flops=0
        )
        dag.add_edge(u=ref, v=node)
        net['nodes'].append(node)
        net['awaiting_nodes'].append(node)

        return True

    @staticmethod
    def back_connect(net):
        """Connect back to a node in a layer specified by int stack"""
        # Do nothing if there aren't enough nodes in either stack or if the shapes aren't the same
        if len(net['nodes']) < 1 or len(net['awaiting_nodes']) < 1 or net['nodes'][0].shape != net['awaiting_nodes'][0].shape:
            return False

        # Add awaiting, node to recurrence dictionary
        awaiting = net['awaiting_nodes'].popleft()
        net['recurrences'][awaiting] = net['nodes'][0]

        return True

    #########################
    ####### Stack Ops #######
    #########################

    @staticmethod
    def dup(dag, net):
        """Duplicate the top node on the node queue"""
        if len(net['nodes']) < 1:
            return False

        id_layer = nn.Identity()

        ref = net['nodes'][0]  # Don't pop node from stack
        node = Node(
            shape=ref.shape,
            layer=ref.layer,
            fn=id_layer,
            desc="Dup",
            flops=0
        )
        dag.add_edge(u=ref, v=node)
        net['nodes'].append(node)

        return True

    @staticmethod
    def identity(dag, net):
        """Identity function on current node. New node will be on the next layer. Allows branches to progress
        asynchronously"""
        if len(net['nodes']) < 1:
            return False

        id_layer = nn.Identity()

        ref = net['nodes'].popleft()  # Pop node from stack
        node = Node(
            shape=ref.shape,
            layer=ref.layer + 1,
            fn=id_layer,
            desc="Identity",
            flops=0
        )
        dag.add_edge(u=ref, v=node)
        net['nodes'].append(node)

        return True

    @staticmethod
    def for_n(stacks, separate_ints):
        """For loop. Pop from sint stack. A closed parentheses means the end of the block"""
        if separate_ints:
            if len(stacks['sint']) < 1:
                return False
            n = stacks['sint'].pop()
        else:
            if len(stacks['int']) < 1:
                return False
            n = stacks['int'].pop()


        block = []
        instruction = "for_n"
        while instruction != '(':
            stacks['exec'].pop()
            if instruction != "for_n":
                block.append(instruction)

        for _ in range(n):
            for instruction in reversed(block): # Reversed since we're adding left to right originally
                stacks['exec'].append(instruction)

        return True

    # @staticmethod
    # def transpose(dag, net):
    #     """Transpose the top node on the node queue"""
    #     if len(net['nodes']) < 1:
    #         return False
    #
    #     ref = net['nodes'].popleft()  # Pop node from stack
    #     node = Node(
    #         shape=ref.shape[::-1],  # Transpose matrix (reverse shapes. Batch not included here so it's fine)
    #         layer=ref.layer + 1,
    #         fn=transpose,
    #         desc="Transpose",
    #         flops=0
    #     )
    #     dag.add_edge(u=ref, v=node)
    #     net['nodes'].append(node)

    #########################
    ##### Normalization #####
    #########################

    @staticmethod
    def layer_norm(dag, net):
        """Layer Normalization"""
        if len(net['nodes']) < 1:
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()

        # Define partial function
        layer_norm = nn.LayerNorm(pop_node.shape)

        # Create new node
        node = Node(
            shape=pop_node.shape,
            layer=pop_node.layer + 1,
            fn=layer_norm,
            desc="Layer Norm",
            flops=0
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node)

    @staticmethod
    def batch_norm(dag, net):
        """Batch Normalization"""
        if len(net['nodes']) < 1:
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()

        if len(pop_node.shape) < 3:
            batch_norm_layer = nn.BatchNorm1d(pop_node.shape[0])
        else:
            batch_norm_layer = nn.BatchNorm2d(pop_node.shape[0])

        # Create new node
        node = Node(
            shape=pop_node.shape,
            layer=pop_node.layer + 1,
            fn=batch_norm_layer,
            desc="Batch Norm",
            flops=0
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        net['nodes'].append(node)

    #########################
    ###### PyTorch Ops ######
    #########################

    @staticmethod
    def process_torch_ops(dag, net, fn, desc):
        """Pop the top 2 tensors from the tensor stack"""
        # Do nothing if there aren't enough tensors in the stack
        if len(net['nodes']) < 1:
            return False

        # TODO: Euclidean norm is idempotent. Any advantage to multiple relu or softmax? Should I assume?
        # Don't allow redundant operations
        if net['nodes'][0].desc == desc:
            return False

        # Pop the top node from the stack
        pop_node = net['nodes'].popleft()

        # Create new node
        node = Node(
            shape=pop_node.shape,
            layer=pop_node.layer + 1,
            fn=fn,
            desc=desc,
            flops=math.prod(pop_node.shape)  # TODO: For now just use the shape. Can expand on this later
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node)

        # Add new node to stack
        net['nodes'].append(node)

        return True

    # Activation Functions
    @staticmethod
    def relu(dag, net):
        """ReLU Activation Function"""
        relu_layer = nn.ReLU()
        Instructions.process_torch_ops(dag, net, relu_layer, "ReLU")

    @staticmethod
    def sigmoid(dag, net):
        """Sigmoid Activation Function"""
        sigmoid_layer = nn.Sigmoid()
        Instructions.process_torch_ops(dag, net, sigmoid_layer, "Sigmoid")

    @staticmethod
    def tanh(dag, net):
        """Tanh Activation Function"""
        tanh_layer = nn.Tanh()
        Instructions.process_torch_ops(dag, net, tanh_layer, "Tanh")

    # @staticmethod
    # def softmax(dag, net):
    #     '''Softmax Activation Function'''
    #     Instructions.process_torch_ops(dag, net, lambda x: torch.softmax(x, dim=1), "Softmax")

    # # Normalization Functions
    # @staticmethod
    # def l2_norm(dag, net):
    #     '''L2 Normalize'''
    #     Instructions.process_torch_ops(dag, net, torch.nn.functional.normalize, "L2 Norm")
    #
    # @staticmethod
    # def batch_norm(dag, net):
    #     '''Batch Normalization'''
