from interpreter import Interpreter
from instructions import Instructions
import torch
import copy
import random

INT_RANGE = (1, 1000)
FLOAT_RANGE = (0.0, 1.0)
OUTPUT_SIZE = 10
INPUT_SIZE = 28
BATCH_SIZE = 64

class Genome:
    '''Genome of a Push Program'''
    def __init__(self):
        self.genome = []
        self.fitness = 0
        self.interpreter = Interpreter()
        self.instructions = Instructions()

    def random_int(self):
        '''Returns a random int'''
        type = random.randint(0, 3)
        match type:
            case 0:
                return random.randint(*INT_RANGE)
            case 1:
                return OUTPUT_SIZE
            case 2:
                return INPUT_SIZE
            case 3:
                return BATCH_SIZE

    def random_index(self):
        '''Returns a random index in the genome'''
        return random.randint(0, len(self.genome))

    def initialize_random(self, num_genes):
        '''Initializes the genome with random genes'''
        # Add genes
        # TODO: Should it add a random number? Or at least allow for that?
        for _ in range(num_genes):
            self.add_gene()

    def random_gene(self):
        '''Returns a random gene'''
        # Randomly select what type of thing to add
        type = random.randint(0, 3)

        # TODO: One to add output shape, another to add input shape
        # TODO: Make function for random int that includes input/output shape as a possibility (one of those or random)
        match type:
            case 0:
                return random.randint(*INT_RANGE)  # Random int
            case 1:
                return random.uniform(*FLOAT_RANGE) # Random float
            case 2:
                return random.choice(list(self.instructions.instructions))  # Add instruction. Project to list for random.choice to work
            case 3:
                return torch.randn(self.random_int(), self.random_int(), requires_grad=True)  # Add tensor

    def evolve(self):
        '''Evolves the genome'''
        # Randomly select what type of thing to do
        type = random.randint(0, 2)

        match type:
            case 0:
                self.add_gene()
            case 1:
                self.remove_gene()
            case 2:
                self.mutate()

    def add_gene(self):
        '''Adds a gene'''
        # Randomly select an index to mutate
        index = self.random_index()
        self.genome.insert(index, self.random_gene())

    def remove_gene(self):
        '''Removes a gene'''
        # Randomly select an index to remove
        index = self.random_index()
        self.genome.pop(index)

    def mutate(self):
        '''Mutates the genome'''
        # Randomly select an index to mutate
        index = self.random_index()
        # Change index to be another random gene
        self.genome[index] = self.random_gene()

    def transcribe(self, train, test):
        '''Transcribes the genome to create a network. Returns the network'''
        self.interpreter.read_genome(self.genome)
        return self.interpreter.create_network(train, test)

class Population:
    '''Population of Push Program Genomes'''
    def __init__(self, size, num_initial_genes):
        self.size = size
        self.population = [Genome() for _ in range(size)]
        for genome in self.population:
            genome.initialize_random(num_initial_genes)

    def forward_generation(self):
        '''Moves the population forward one generation'''
        # Sort the population by fitness. Right now fitness is loss, so lower is better
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        # Copy the top 3 to the bottom 3
        self.population[:3] = copy.deepcopy(self.population[-3:])
        # Mutate the copied top 3
        for genome in self.population[:3]:
            genome.evolve()

    def run(self, generations, train, test):
        '''Runs the population on the train and test data'''
        for _ in range(generations):
            for genome in self.population:
                network = genome.transcribe(train, test)
                # Train the network
                genome.fitness = network.fit()
                if genome.fitness != float('inf'):
                    print(genome.genome)
