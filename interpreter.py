import torch
from instructions import Instructions
from dag import *

class Interpreter:
    '''Push Interpreter'''
    def __init__(self, input_shape, output_shape):
        self.stacks = {
            'int': [], # Really just Natural numbers
            'float': [],
            'bool': [],
            'node': [],
            'str': [],
            'exec': [], # Contains the instructions

            'params': [], # Parameter stack for optimization
        }

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.instructions = Instructions()

    def read_genome(self, genome):
        '''Reads the genome and processes it into stacks'''
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
        '''Runs the Program. Generates Computation Graph'''
        root = Node(self.input_shape, 0, None) # Create root node with input shape, no function
        dag = DAG(root)
        self.stacks['node'].append(root)

        # TODO: Could do fn = identity
        while len(self.stacks['exec']) > 0:
            # Get next instruction
            instr = self.stacks['exec'].pop(0)
            # Execute instruction
            self.instructions(dag, self.stacks, instr)
            # Interpreter reads an instruction. Instructions executes said instruction, which should create a node in the DAG
            # TODO: When we first construct the graph, use references to the tensors in the stack. The instructions give a lambda argument
            # TODO: To the graph