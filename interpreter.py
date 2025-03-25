import utils
import gp
from instructions import Instructions
from dag import *
from functions import *
from network import Network
import torch.nn.init as init
import torch.nn as nn
import copy

# Functions to be activated if activation is not None (default is relu)
ACTIVE = ['matmul', 'conv2d']
# Functions to have manual bias added. Conv2d does it automatically so don't include it here.
BIAS = ['matmul']

class Interpreter:
    """
    Push Interpreter. By default, automatically adds relu and bias where applicable and
    separates small vs. large integers
    """
    def __init__(self, input_shape, output_shape, activation='relu', auto_bias=True, separate_ints=True,
                 embedding=None, embed_dim=None, vocab_size=None, recurrent=False):
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
            'awaiting_nodes': deque([]), # Queue holding nodes that are waiting for a backwards connection
            'recurrences': {}, # Dictionary holding recurrences
        }

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Network parameters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.auto_bias = auto_bias
        self.separate_ints = separate_ints

        # TODO: Use premade embeddings so just remove all of this? Unsure
        # Embedding parameters
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Recurrent parameters
        self.recurrent = recurrent

        # Initialize instructions
        self.instructions = Instructions(activation=self.activation)

    def read_genome(self, genome):
        """Reads genome into exec stack"""
        self.stacks['exec'] = copy.deepcopy(genome)

    def read_instr(self, instr, dag):
        """Reads an instruction and processes it accordingly"""
        if type(instr) == int:
            if self.separate_ints:
                if instr <= gp.SINT_RANGE[1]:
                    self.stacks['sint'].append(instr)
                else:
                    self.stacks['int'].append(instr)
            else:
                self.stacks['int'].append(instr)
        elif type(instr) == float:
            self.stacks['float'].append(instr)
        elif type(instr) == bool:
            self.stacks['bool'].append(instr)
        elif instr in self.instructions.instructions:
            return self.instructions(dag, self.net, self.stacks, self.device, instr, self.separate_ints)
        elif type(instr) == str:
            if instr =='(': # Denotes end of a function
                return True
            else:
                self.stacks['str'].append(instr)

        return True

    def run(self):
        """Runs the program. Generates computation graph, prunes unnecessary nodes, and returns network"""
        id_layer = nn.Identity()
        root = Node(shape=self.input_shape, layer=0, fn=id_layer) # Create root node with input shape, no function
        dag = DAG(root)
        self.net['nodes'].append(root)

        if self.embedding is not None:
            self.add_embedding(dag)

        # Graph should be created after this
        while len(self.stacks['exec']) > 0:
            # Get next instruction
            instr = self.stacks['exec'].pop()

            # Execute instruction
            added = self.read_instr(instr, dag)

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

        self.add_output(dag) # Add output layer

        # Create network
        network = Network(
            dag=dag,
            root=root,
            recurrences=self.net['recurrences'],
            device=self.device,
        )
        return network

    def clear(self):
        """Clear current data in interpreter"""
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
            'awaiting_nodes': deque([]),
            'recurrences': {},
        }

    def add_embedding(self, dag):
        """Adds an embedding layer to the input"""
        last_node = self.net['nodes'].popleft()
        last_shape = last_node.shape

        # TODO: use self.embedding as the fn (so some kind of embedding module)
        node = Node(
            shape=(last_shape[0], self.embed_dim),
            layer=1, # Since this will only ever come after the root
            fn=embedding,
            desc="Embedding",
            flops=self.input_shape[0] * self.embed_dim,  # Approximate cost
        )

        dag.add_edge(last_node, node)
        self.net['nodes'].append(node)

    def add_output(self, dag):
        """Adds the output layer to the DAG, There should always be at least 1 node in the stack"""
        # Get the last node in the stack
        last_node = self.net['nodes'].popleft()
        last_shape = last_node.shape
        last_dim, output_dim = len(last_shape), len(self.output_shape)

        if last_dim < output_dim:
            # TODO: Fix this
            # If last_dim < output_dim, we need project it up to the output dim.
            for _ in range(output_dim - last_dim):
                # Add a node that unsqueezes the last dimension to the dag.
                node = Node(
                    shape=last_shape + (1,),
                    layer=last_node.layer + 1,
                    fn=torch.unsqueeze,
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

            flatten_layer = nn.Flatten()

            node = Node(
                shape=(prod,),
                layer=last_node.layer + 1,
                fn=flatten_layer,
                desc="Flatten",
                flops=0
            )
            dag.add_edge(last_node, node)
            last_node = node

        last_shape = last_node.shape

        # Calculate flops depending on dimension
        if len(last_node.shape) < 2:
            flops = (2 * last_node.shape[-1] - 1) * self.output_shape[-1]
        else:
            flops = (2 * last_node.shape[-1] * last_node.shape[-2] - 1) * self.output_shape[-1]

        matmul_layer = nn.Linear(last_shape[-1], self.output_shape[-1], bias=False)

        # Add node that projects to the output shape. Need matrix. Result should be output_shape[-1]
        node = Node(
            shape=utils.mult_shape(last_node.shape, (last_shape[-1], self.output_shape[-1])),
            layer=last_node.layer + 1,
            fn=matmul_layer,
            desc="Matmul",
            flops=flops
        )

        dag.add_edge(last_node, node)

        # Prune all nodes that aren't in the path of the output layer. This is crucial. Forward pass can fail if we don't do this.
        # Due to different branches. There has to be only one final output when we conduct a forward pass
        dag.prune(node)

        # TODO: Support for activation functions?