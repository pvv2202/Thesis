import torch
import utils
import gp
from collections import deque
from instructions import Instructions
from dag import *
from utils import *
from functions import *
from network import Network
import torch.nn.init as init

# TODO: Maybe do the same with bias?
# Functions to be activated if activation is not None (default is relu)
ACTIVE = ['matmul', 'conv2d']
# Functions to have manual bias added. Conv2d does it automatically so don't include it here.
BIAS = ['matmul']

class Interpreter:
    '''
    Push Interpreter. By default, automatically adds relu and bias where applicable and
    separates small vs. large integers
    '''
    def __init__(self, train, test, activation='relu', auto_bias=True, separate_ints=True):
        self.stacks = {
            'int': [], # Really just Natural numbers
            'sint': [], # Small integers
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
        self.auto_bias = auto_bias
        self.separate_ints = separate_ints

        # Get input/output shapes
        train_x, train_y = next(iter(train)) # Get example input/output
        self.input_shape = tuple(train_x.size()[1:])
        if train_y.ndim == 0: # Single value
            self.output_shape = (1,)
        elif train_y.ndim == 1: # Classification
            num_classes = len(torch.unique(train_y))
            self.output_shape = (num_classes,)
            # self.output_shape = (10,)
        else: # Regression or multi-class/multi-label classification
            self.output_shape = tuple(train_y.size())

        # Initialize instructions
        self.instructions = Instructions(activation=self.activation)

    def read_genome(self, genome):
        '''Reads the genome and processes it into stacks'''
        print(genome)
        for gene in genome:
            if type(gene) == int:
                if self.separate_ints:
                    if gene <= gp.SINT_RANGE[1]:
                        self.stacks['sint'].append(gene)
                    else:
                        self.stacks['int'].append(gene)
                else:
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
            added = self.instructions(dag, self.net, self.stacks, self.device, instr, self.separate_ints)

            # Add bias, activation if specified. Rotate because we can have multiple branches.
            # The added node will always be the last element and we need to make sure bias and activation operate on that
            if added:
                # If auto bias is not None and instruction requires bias, add bias (mat_add) to stack
                if self.auto_bias is not None and instr in BIAS:
                    self.net['nodes'].rotate(1)
                    self.instructions(dag, self.net, self.stacks, self.device, 'mat_add', self.separate_ints)
                # If activation is not None and instruction requires activation, add activation function to stack
                if self.activation is not None and instr in ACTIVE:
                    self.net['nodes'].rotate(1)
                    self.instructions(dag, self.net, self.stacks, self.device, self.activation, self.separate_ints)


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

    def clear(self):
        '''Clear current data in interpreter'''
        self.stacks = {
            'int': [],
            'sint': [],
            'float': [],
            'bool': [],
            'str': [],
            'exec': [],
        }
        self.net = {
            'nodes': deque([]),
            'params': []
        }

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
                    desc="Unsqueeze",
                    flops=0
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
                fn=flatten,
                parents=[last_node],
                desc="Flatten",
                flops=0
            )
            dag.add_edge(last_node, node)
            last_node = node

        last_shape = last_node.shape

        # Add node that projects to the output shape. Need matrix. Result should be output_shape[-1]
        weights = torch.empty(last_shape[-1], self.output_shape[-1], requires_grad=True, device=self.device)
        init.xavier_uniform_(weights)
        self.net['params'].append(weights)

        # Calculate flops depending on dimension
        if len(last_node.shape) < 2:
            flops = (2 * last_node.shape[-1] - 1) * weights.shape[-1]
        else:
            flops = (2 * last_node.shape[-1] * last_node.shape[-2] - 1) * weights.shape[-1]

        node = Node(
            shape=utils.mult_shape(last_node.shape, weights.shape),
            layer=last_node.layer + 1,
            fn=matmul,
            parents=[last_node],
            weight_id=len(self.net['params'])-1,
            desc="Matmul",
            flops=flops
        )

        dag.add_edge(last_node, node)

        # Prune all nodes that aren't in the path of the output layer. This is crucial. Forward pass can fail if we don't do this.
        # Due to different branches. There has to be only one final output when we conduct a forward pass
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