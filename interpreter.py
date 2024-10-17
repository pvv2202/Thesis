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

            'params': [] # Parameter stack for optimization
        }

        # Initialize instructions
        self.instructions = Instructions()

    # TODO: Need to run the program and then use params as push_tensor. They are added in order, so we should be able to use them in order
    # TODO: Could just replace push_tensor with a modified version that takes them from params
    def read_genome(self, genome):
        '''Reads the genome, processes it into stacks, and creates a network'''
        # Process genome into stacks
        for instruction in genome:
            if type(instruction) == int:
                self.stacks['int'].append(instruction)
            elif type(instruction) == float:
                self.stacks['float'].append(instruction)
            elif type(instruction) == bool:
                self.stacks['bool'].append(instruction)
            elif instruction in self.instructions.instructions:
                self.stacks['exec'].append(instruction)
            elif type(instruction) == str:
                self.stacks['str'].append(instruction)

    def create_network(self, train, test):
        '''Creates a network and accepts train/test data to shape the input/output accordingly'''
        return Network(self.stacks, train, test)

    # TODO: Add create_network function that creates a network and accepts train/test data to shape the input/output accordingly