import Interpreter
import random

class Genome():
    '''Genome of a Neural Network'''
    INT_RANGE = [1, 200]
    FLOAT_RANGE = [0.0, 1.0]

    def __init__(self):
        self.genome = ['input_layer', 'output_layer', 'compile', 'fit']
        self.fitness = 0

    def mutate(self):
        '''Mutates the genome'''

        # Randomly select an index to mutate
        index = random.randint(1, len(self.genome) - 3)

        # Randomly select what type of thing to add
        type = random.randint(0, 3)
        match type:
            case 0:
                self.genome.insert(index, random.randint(*INT_RANGE))
            case 1:
                self.genome.insert(index, 'dense')
            case 2:
                self.genome.insert(index, 'dropout')
            case 3:
                self.genome.insert(index, 'normalize')

class Population():
    '''Population of Neural Networks'''
    def __init__(self, size=10):
        self.population = []
        self.fitness = [] # Each index corresponds to the fitness of the network at that index
        self.size = size

    def initialize_population(self):
        '''Initialize the population with random neural networks'''
        for i in range(self.size):
            self.population.append(Genome())

if __name__ == '__main__':
    interpreter = Interpreter.PushInterpreter()
    interpreter.exec_stack.extend([
        interpreter.input_layer,
        interpreter.conv,
        interpreter.conv,
        interpreter.max_pool,
        interpreter.normalize,
        interpreter.conv,
        interpreter.max_pool,
        interpreter.normalize,
        interpreter.global_pool,
        interpreter.output_layer,
        interpreter.compile,
        interpreter.fit
    ])
    interpreter.str_stack.extend([
        'relu',
        'relu',
        'relu',
        'relu',
    ])
    interpreter.run()