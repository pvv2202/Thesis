import torch
import utils
from collections import deque
from instructions import Instructions
from dag import *
from utils import *
from network import Network

# TODO: Maybe do the same with bias?
# Functions to be activated if activation is not None (default is relu)
ACTIVE = ['matmul', 'conv2d']

class Interpreter:
    '''Push Interpreter'''
    def __init__(self, train, test, activation):
        self.stacks = {
            'int': [], # Really just Natural numbers
            'float': [],
            'bool': [],
            'str': [],
            'exec': [], # Contains the instructions
        }

        # Network structures
        self.net = {
            'nodes': deque([]), # Queue holding nodes added to the graph
            'params': [] # Parameter stack for PyTorch autograd
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train = train
        self.test = test
        # TODO: Improve this. Right now you can specify a consistent activation function (default relu).
        self.activation = activation

        # Get input/output shapes
        train_x, train_y = next(iter(train)) # Get example input/output
        self.input_shape = tuple(train_x.size()[1:])
        batch_size = self.input_shape[0]
        if train_y.ndim == 0: # Single value
            self.output_shape = (1,)
        elif train_y.ndim == 1: # Classification
            # num_classes = len(torch.unique(train_y))
            # self.output_shape = (num_classes,)
            self.output_shape = (10,)
        else: # Regression or multi-class/multi-label classification
            self.output_shape = tuple(train_y.size())

        # Initialize instructions
        self.instructions = Instructions(activation=self.activation)

    def read_genome(self, genome):
        '''Reads the genome and processes it into stacks'''
        print(genome)
        for gene in genome:
            if type(gene) == int:
                self.stacks['int'].append(gene)
            elif type(gene) == float:
                self.stacks['float'].append(gene)
            elif type(gene) == bool:
                self.stacks['bool'].append(gene)
            elif gene in self.instructions.instructions:
                self.stacks['exec'].append(gene)
            elif type(gene) == str:
                self.stacks['str'].append(gene)

    def run(self):
        '''Runs the program. Generates computation graph, prunes unnecessary nodes, and returns network'''
        root = Node(shape=self.input_shape, layer=0, fn=None) # Create root node with input shape, no function
        dag = DAG(root)
        self.net['nodes'].append(root)

        # Graph should be created after this
        while len(self.stacks['exec']) > 0:
            # Get next instruction
            instr = self.stacks['exec'].pop()
            # Execute instruction
            added = self.instructions(dag, self.net, self.stacks, self.device, instr)
            # If activation is not None and instruction requires activation, add activation function to stack
            if added and self.activation is not None and instr in ACTIVE:
                self.instructions(dag, self.net, self.stacks, self.device, self.activation)

            # if self.net['nodes'][-1].shape == None:
            #     print(self.net['nodes'][-1].desc)
            #     for parent in self.net['nodes'][-1].parents:
            #         print(parent.shape)

        self.add_output(dag) # Add output layer

        # Create network
        network = Network(
            dag=dag,
            train=self.train,
            test=self.test,
            params=self.net['params'],
            device=self.device
        )
        return network

    def add_output(self, dag):
        '''Adds the output layer to the DAG, There should always be at least 1 node in the stack'''
        # Get the last node in the stack
        last_node = self.net['nodes'].popleft()
        last_shape = last_node.shape
        last_dim, output_dim = len(last_shape), len(self.output_shape)

        if last_dim < output_dim:
            # If last_dim < output_dim, we need project it up to the output dim.
            for _ in range(output_dim - last_dim):
                # Add a node that unsqueezes the last dimension to the dag.
                node = Node(
                    shape=last_shape + (1,),
                    layer=last_node.layer + 1,
                    fn=torch.unsqueeze,
                    parents=[last_node],
                    desc="Unsqueeze"
                )
                dag.add_edge(last_node, node)
                last_node = node
        elif last_dim > output_dim:
            # If last_dim > output_dim, Add a node that flattens the last dimension. Flatten ignores the batch dimension.
            prod = 1
            for x in last_shape:
                prod *= x

            node = Node(
                shape=(prod,),
                layer=last_node.layer + 1,
                fn=lambda x: torch.flatten(x, start_dim=1),
                parents=[last_node],
                desc="Flatten"
            )
            dag.add_edge(last_node, node)
            last_node = node

        last_shape = last_node.shape

        # Add node that projects to the output shape. Need matrix. Result should be output_shape[-1]
        weights = torch.randn(last_shape[-1], self.output_shape[-1], requires_grad=True, device=self.device)
        self.net['params'].append(weights)

        node = Node(
            shape=utils.mult_shape(last_node.shape, weights.shape),
            layer=last_node.layer + 1,
            fn=torch.matmul,
            parents=[last_node],
            weight_id=len(self.net['params'])-1,
            desc="Matmul"
        )

        dag.add_edge(last_node, node)

        # Prune all nodes that aren't in the path of the output layer. This is crucial. Forward pass can fail if we don't do this.
        dag.prune(node)

        # TODO: Add support for activation functions
        # last_node = node

        # # Add activation function
        # node = Node(
        #     shape=self.output_shape,
        #     layer=last_node.layer + 1,
        #     fn=lambda x: torch.softmax(x, dim=1), # TODO: Need to update this. This assumes softmax but we should support other activation functions
        #     parents=[last_node]
        # )

        # dag.add_edge(last_node, node)
        # return dag

        # We need the dimensions of the resulting matrix to = self.output_shape. This depends on the loss.
        # Generally speaking, should probably be batch size x num_features.