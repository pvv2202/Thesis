import copy
import torch
from instructions import Instructions
from network import Network

class Interpreter:
    '''Push Interpreter'''
    def __init__(self):
        # Initialize stacks
        self.stacks = {
            'int': [], # Really just Natural numbers
            'float': [],
            'bool': [],
            'str': [],
            'tensor': [],
            'exec': [],

            'params': [], # Parameter stack for optimization
        }

        # Initialize instructions
        self.instructions = Instructions()

    # TODO: Need to run the program and then use params as push_tensor. They are added in order, so we should be able to use them in order
    # TODO: Could just replace push_tensor with a modified version that takes them from params
    def read_genome(self, genome):
        '''Reads the genome and processes it into stacks'''
        # Process genome into stacks
        for gene in genome:
            if type(gene) == int:
                self.stacks['int'].append(gene)
            elif type(gene) == float:
                self.stacks['float'].append(gene)
            elif type(gene) == bool:
                self.stacks['bool'].append(gene)
            elif type(gene) == torch.Tensor:
                self.stacks['tensor'].append(gene)
                self.stacks['params'].append(gene)
            elif gene in self.instructions.instructions:
                self.stacks['exec'].append(gene)
            elif type(gene) == str:
                self.stacks['str'].append(gene)

    def create_network(self, train, test):
        '''Creates a network from the stacks'''
        stack_copy = copy.deepcopy(self.stacks)
        return Network(stack_copy, train, test)

    # TODO: Add create_network function that creates a network and accepts train/test data to shape the input/output accordingly