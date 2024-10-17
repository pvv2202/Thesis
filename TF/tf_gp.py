import tf_interpreter
import random

INT_RANGE = (1, 112)
FLOAT_RANGE = (0.0, 1.0)

class Genome():
    '''Genome of a Neural Network'''

    def __init__(self):
        self.genes = ['input_layer', 'output_layer', 'compile', 'fit', 'evaluate']
        self.fitness = 0

    def add_gene(self):
        '''Adds a gene'''
        # Randomly select an index to mutate
        index = random.randint(1, len(self.genes) - 4)
        # Randomly select what type of thing to add
        type = random.randint(0, 3)

        match type:
            case 0:
                self.genes.insert(index, random.randint(*INT_RANGE))
            case 1:
                self.genes.insert(index, random.uniform(*FLOAT_RANGE))
            case 2:
                self.genes.insert(index, random.choice(tf_interpreter.valid_instruction_mutations))
            case 3:
                self.genes.insert(index, random.choice(tf_interpreter.valid_activation_mutations))

    def remove_gene(self):
        '''Removes a gene'''
        # Randomly select an index to mutate
        index = random.randint(1, len(self.genes) - 3)

        # Check that the current index is not a required gene
        if self.genes[index] not in {'input_layer', 'output_layer', 'compile', 'fit'}:
            self.genes.pop(index)

    def mutate(self):
        '''Mutates the genome'''
        # Randomly select an index to mutate
        index = random.randint(1, len(self.genes) - 3)

        # Check that the current index is not a required gene. If it is just return
        if self.genes[index] in {'input_layer', 'output_layer', 'compile', 'fit'}:
            return

        # Randomly select what type of thing to mutate
        type = random.randint(0, 3)
        match type:
            case 0:
                self.genes[index] = random.randint(*INT_RANGE)
            case 1:
                self.genes[index] = random.uniform(*FLOAT_RANGE)
            case 2:
                self.genes[index] = random.choice(tf_interpreter.valid_instruction_mutations)
            case 3:
                self.genes[index] = random.choice(tf_interpreter.valid_activation_mutations)

    def transcribe(self, X_train, y_train, X_test, y_test):
        '''Transcribes the genome'''
        interpreter = tf_interpreter.TFInterpreter(X_train, y_train, X_test, y_test)
        interpreter.read_genome(self.genes)
        score = interpreter.run()
        self.fitness = score[0]

class Population():
    '''Population of Neural Networks'''
    def __init__(self, size=10):
        self.population = []
        self.fitness = [] # Each index corresponds to the fitness of the network at that index
        self.size = size

    def initialize_population(self, gene_number=10):
        '''Initialize the population with random neural networks'''
        for i in range(self.size):
            self.population.append(Genome())
            # Add 10 initial genes
            for _ in range(gene_number):
                self.population[i].add_gene()

    def forward_generation(self):
        '''Move forward one generation'''
        # Remove the bottom 3 from the population
        self.population.sort(key=lambda x: x.fitness)

        # Clone best 3
        self.population[-3:] = self.population[:3]

        # Mutate the rest. Add a gene with 35% probability, remove a gene with 15% probability, mutate a gene with 50% probability
        for genome in self.population[1:]:
            rand = random.random()
            num = random.randint(1, 3)
            if rand < 0.35:
                for _ in range(num):
                    genome.add_gene()
            elif rand < 0.5:
                for _ in range(num):
                    genome.remove_gene()
            else:
                for _ in range(num):
                    genome.mutate()

    def run(self, X_train, y_train, X_test, y_test, generations=100):
        '''Runs the genetic algorithm'''
        self.initialize_population()
        for _ in range(generations):
            for genome in self.population:
                genome.transcribe(X_train, y_train, X_test, y_test)
            self.forward_generation()
            print(self.population[0].fitness)
            print(self.population[0].genome)
            print()