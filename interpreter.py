import utils
import gp
from instructions import Instructions
from dag import *
from functions import *
from network import Network
import torch.nn.init as init

# Functions to be activated if activation is not None (default is relu)
ACTIVE = ['matmul', 'conv2d']
# Functions to have manual bias added. Conv2d does it automatically so don't include it here.
BIAS = ['matmul']

class Interpreter:
    '''
    Push Interpreter. By default, automatically adds relu and bias where applicable and
    separates small vs. large integers
    '''
    def __init__(self, input_shape, output_shape, activation='relu', auto_bias=True, separate_ints=True, embedding=False, embed_dim=None, vocab_size=None, recurrent=False):
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

        # Network parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.auto_bias = auto_bias
        self.separate_ints = separate_ints

        # Embedding parameters
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Recurrent parameters
        self.recurrent = recurrent

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

        if self.embedding:
            self.add_embedding(dag)

        if self.recurrent:
            self.add_hidden_state(dag)

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
            params=self.net['params'],
            device=self.device,
            recurrent=self.recurrent
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

    def add_hidden_state(self, dag):
        last_node = self.net['nodes'][0]  # Don't pop so we can use it with this node or others

        # Add recurrent layer
        hidden_node = Node(
            shape=(self.output_shape),
            layer=last_node.layer,
            fn=None,
            parents=[self.net['nodes'][0]],
            desc="Hidden State",
            flops=0,
            weight_id=None
        )

        dag.hidden_node = hidden_node  # Add hidden node to graph

        dag.add_edge(last_node, hidden_node)
        self.net['nodes'].append(hidden_node)

    def add_embedding(self, dag):
        '''Adds an embedding layer to the input'''
        last_node = self.net['nodes'].popleft()
        last_shape = last_node.shape

        weights = torch.empty(self.vocab_size, self.embed_dim, requires_grad=True, device=self.device)
        init.xavier_uniform_(weights)
        self.net['params'].append(weights)

        node = Node(
            shape=(last_shape[0], self.embed_dim),
            layer=1, # Since this will only ever come after the root
            fn=embedding,
            parents=[last_node],
            desc="Embedding",
            flops=self.input_shape[0] * self.embed_dim,  # Approximate cost
            weight_id=len(self.net['params'])-1
        )

        dag.add_edge(last_node, node)
        self.net['nodes'].append(node)

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