import torch

import utils
from dag import *
from utils import *
import inspect

# TODO: Note that when I make GP, I need to exclude instructions with "process" in the name

class Instructions:
    '''Instructions for the Push Interpreter'''
    def __init__(self):
        '''Initialize Instructions'''
        self.instructions  = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]

    def __call__(self, dag, stacks, instruction):
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
        return method(**kwargs)

    # TODO: Add convolution. Will need to add some functions to utils as well presumably

    #########################
    # Matrix Multiplication #
    #########################

    @staticmethod
    def matmul(dag, stacks):
        '''Matrix Multiplication'''
        # Do nothing if there are no nodes or integers
        if len(stacks['node']) < 1 or len(stacks['int']) < 1:
            return

        # Pop the top node and integer from the stack
        pop_node = stacks['node'].pop()
        pop_int = stacks['int'].pop()

        # Calculate new dimension
        pop_dim = pop_node.dim
        if len(pop_dim) < 2: # If less than 2, just make them the same (i.e. 1D tensor)
            new_dim = pop_dim
        else: # Otherwise, make the second to last dimension the same as the popped node's last dimension
            new_dim = (pop_dim[-2], pop_int)

        # Create weights
        weights = torch.randn(new_dim, requires_grad=True)
        stacks['params'].append(weights) # Add weights to the parameters stack

        # Create new node with the output shape of the matrix multiplication
        node = Node(
            shape=utils.mult_shape(pop_dim, new_dim),
            layer=pop_int.layer + 1,
            fn=torch.matmul,
            parents=[pop_node],
            weights=weights
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node, layer=pop_int.layer + 1)

        # Add new node to stack
        stacks['node'].append(node)

    @staticmethod
    def matmul_dup(dag, stacks):
        '''Matrix Multiplication with Duplicate'''
        # Get the top node from the stack
        pop_node = stacks['node'][-1]
        # Call matmul
        Instructions.matmul(dag, stacks)
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
            fn=torch.matmul,
            parents=[pop_node1, pop_node2]
        )
        dag.add_edge(u=pop_node1, v=node, layer=node.layer) # Edge between node1 and new node
        dag.add_edge(u=pop_node2, v=node, layer=node.layer) # Edge between node2 and new node

    #########################
    #### Matrix Addition ####
    #########################

    @staticmethod
    def mat_add(dag, stacks):
        '''Matrix Addition'''
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
            fn=torch.add,
            parents=[pop_node1, pop_node2]
        )
        dag.add_edge(u=pop_node1, v=node, layer=node.layer)

    @staticmethod
    def mat_add_stack(dag, stacks):
        '''Matrix Addition'''
        # Do nothing if there aren't enough nodes in the stack
        if len(stacks['node']) < 1:
            return

        # Pop the top node from the stack
        pop_node = stacks['node'].pop()

        # Create weights of same shape as popped node
        weights = torch.randn(pop_node.shape, requires_grad=True)
        stacks['params'].append(weights) # Add weights to the parameters stack

        # Create new node
        node = Node (
            shape=pop_node.shap, # Shape is the same as the popped node
            layer=pop_node.layer + 1, # Take the max layer of the two nodes and add 1
            fn=torch.add,
            parents=[pop_node],
            weights=weights
        )
        dag.add_edge(u=pop_node, v=node, layer=node.layer) # Edge between node1 and new node

    @staticmethod
    def mat_add_dup(dag, stacks):
        '''Matrix Addition with Duplicate'''
        # Get the top node from the stack
        pop_node = stacks['node'][-1]
        # Call mat_add
        Instructions.mat_add(dag, stacks)
        # Add top node back
        stacks['node'].append(pop_node)

    #########################
    ###### PyTorch Ops ######
    #########################

    @staticmethod
    def process_torch_ops(dag, stacks, fn):
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
            fn=fn,
            parents=[pop_node]
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node, layer=node.layer)

        # Add new node to stack
        stacks['node'].append(node)

    # Activation Functions
    @staticmethod
    def relu(dag, stacks):
        '''ReLU Activation Function'''
        Instructions.process_torch_ops(dag, stacks, torch.relu)

    @staticmethod
    def sigmoid(dag, stacks):
        '''Sigmoid Activation Function'''
        Instructions.process_torch_ops(dag, stacks, torch.sigmoid)

    @staticmethod
    def softmax(dag, stacks):
        '''Softmax Activation Function'''
        Instructions.process_torch_ops(dag, stacks, torch.softmax)

    # Others
    @staticmethod
    def normalize(dag, stacks):
        '''Normalize'''
        Instructions.process_torch_ops(dag, stacks, torch.normalize)

    #########################
    ### Matrix Scalar Ops ###
    #########################
    @staticmethod
    def process_mat_scalar_ops(dag, stacks, fn):
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
            parents=[pop_node]
        )

        # Add the new node to the graph
        dag.add_edge(u=pop_node, v=node, layer=node.layer)

        # Add new node to stack
        stacks['node'].append(node)

    @staticmethod
    def mat_add_int(dag, stacks):
        '''Matrix Addition with Int'''
        # Do nothing if there aren't enough ints in the stack
        if len(stacks['int']) < 1:
            return

        pop_int = stacks['int'].pop()
        Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.add(x, pop_int))

    @staticmethod
    def mat_add_float(dag, stacks):
        '''Matrix Addition with Float'''
        # Do nothing if there aren't enough floats in the stack
        if len(stacks['float']) < 1:
            return

        pop_float = stacks['float'].pop()
        Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.add(x, pop_float))

    @staticmethod
    def mat_mult_float(dag, stacks):
        '''Matrix Multiplication with Float'''
        # Do nothing if there aren't enough floats in the stack
        if len(stacks['float']) < 1:
            return

        pop_float = stacks['float'].pop()
        Instructions.process_mat_scalar_ops(dag, stacks, lambda x: torch.mul(x, pop_float))

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

    @staticmethod
    def add_int(stacks):
        '''Add ints from int stack'''
        Instructions.process_ints(stacks, lambda x, y: x + y)

    @staticmethod
    def mult_int(stacks):
        '''Multiply ints from int stack'''
        Instructions.process_ints(stacks, lambda x, y: x * y)

    #########################
    ####### Float Ops #######
    #########################

    @staticmethod
    def process_floats(stacks, fn):
        '''Pop the top 2 floats from the float stack'''
        # Do nothing if there aren't enough floats in the stack
        if len(stacks['float']) < 2:
            return

        fn(stacks['float'].pop(), stacks['float'].pop())

    @staticmethod
    def add_float(stacks):
        '''Add floats from float stack'''
        Instructions.process_floats(stacks, lambda x, y: x + y)

    @staticmethod
    def sub_float(stacks):
        '''Subtract floats from float stack'''
        Instructions.process_floats(stacks, lambda x, y: x - y)

    @staticmethod
    def mult_float(stacks):
        '''Multiply floats from float stack'''
        Instructions.process_floats(stacks, lambda x, y: x * y)

    @staticmethod
    def div_float(stacks):
        '''Divide floats from float stack'''
        Instructions.process_floats(stacks, lambda x, y: x / y)