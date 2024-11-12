from interpreter import Interpreter
from instructions import Instructions
import matplotlib.pyplot as plt
import random
import torch
import copy

PRIMES = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23] # Fundamental Theorem of Arithmetic
FLOAT_RANGE = (0.0, 1.0)
ADD_RATE = 0.18
REMOVE_RATE = 1/(1 + ADD_RATE)

# TODO: Just in general, is there a better way to go about the train/test thing? Like could I just give it to the network?
# TODO: Track sizes over time. Usually bad to do it during runs
# TODO: Use UMAD (uniform mutation by addition and deletion). Deletion rate should be addRate/(1 + addRate). Do 2 sweeps. First adds on either side of each index, second deletes at each index
# TODO: Different rates can change a lot

class Genome:
    '''Genome of a Push Program'''
    def __init__(self, train, test, activation):
        self.genome = []
        self.fitness = 0
        self.train = train
        self.test = test
        self.network = None

        # Initialize interpreter, instructions
        self.interpreter = Interpreter(train, test, activation) # Pass shapes to interpreter
        self.instructions = Instructions()

    def random_index(self):
        '''Returns a random index in the genome'''
        return random.randint(0, len(self.genome))

    def initialize_random(self, num_genes):
        '''Initializes the genome with random genes'''
        # Add genes
        for _ in range(num_genes):
            self.genome.append(self.random_gene())

    def random_gene(self):
        '''Returns a random gene'''
        # Randomly select what type of thing to add
        type = random.randint(0, 2)

        match type:
            case 0:
                return random.choice(PRIMES)  # Random int
            case 1:
                return random.choice(self.instructions.instructions)  # Add instruction. Project to list for random.choice to work
            # case 2:
            #     return random.uniform(*FLOAT_RANGE) # Random float

    def UMAD(self):
        '''
        Mutates the genome using UMAD. With some add probability, add a gene before or after each gene. Loop
        through genome again. With remove probability = add probability/(1 + add probability), remove a gene
        '''
        # Add genes
        i = 0
        while i < len(self.genome):
            if random.random() <= ADD_RATE:
                i += random.choice([0, 1])
                self.genome.insert(i, self.random_gene())
            i += 1

        # Remove genes
        i = 0
        while i < len(self.genome):
            if random.random() <= REMOVE_RATE:
                self.genome.pop(i)
            else:
                i += 1 # Only increment i if we didn't remove

    def transcribe(self):
        '''Transcribes the genome to create a network. Returns the network'''
        self.interpreter.read_genome(self.genome)
        self.network = self.interpreter.run()
        return self.network

class Population:
    '''Population of Push Program Genomes'''
    def __init__(self, size, num_initial_genes, train, test, activation=torch.softmax):
        self.size = size
        self.population = [Genome(train, test, activation) for _ in range(size)]
        # Initialize the population with random genes
        for genome in self.population:
            genome.initialize_random(num_initial_genes)

    def forward_generation(self):
        '''Moves the population forward one generation'''
        # Sort the population by fitness. Right now fitness is loss, so lower is better
        self.population.sort(key=lambda x: x.fitness)
        # Replace bottom 2 with copies of top 2
        self.population[:3] = copy.deepcopy(self.population[-2:])
        # Mutate the copied top 3
        for genome in self.population[:-2]:
            genome.UMAD()

    def run(self, generations, epochs):
        '''Runs the population on the train and test data'''
        acc = []
        size = []
        for gen_num in range(1, generations + 1):
            gen_acc = []
            gen_size = []
            for genome in self.population:
                gen_size.append(len(genome.genome))
                network = genome.transcribe()
                print(network)
                # Train the network
                genome.fitness = network.fit(epochs=epochs)
                gen_acc.append(genome.fitness)
                print(f"Genome fitness: {genome.fitness}")
            acc.append(gen_acc)
            size.append(gen_size)

            print(f"Generation {gen_num} completed.")

        # Generate labels for each generation
        labels = [i for i in range(1, generations + 1)]

        # Create box plot for size
        size_plot = plt.boxplot(size,
                          vert=True,
                          patch_artist=True,
                          labels=labels,
                          showmeans=True,
                          meanprops=dict(marker='.', markerfacecolor='black', markeredgecolor='black'),
                          medianprops=dict(color='blue'),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'),
                          boxprops=dict(facecolor='lavender', color='black'),
                          flierprops=dict(markerfacecolor='green', marker='D'))

        plt.title('Box Plot of Size Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Size (Number of Genes)')
        plt.show()

        # Create box plot for accuracy
        acc_plot = plt.boxplot(acc,
                          vert=True,
                          patch_artist=True,
                          labels=labels,
                          showmeans=True,
                          meanprops=dict(marker='.', markerfacecolor='black', markeredgecolor='black'),
                          medianprops=dict(color='blue'),
                          whiskerprops=dict(color='black'),
                          capprops=dict(color='black'),
                          boxprops=dict(facecolor='lavender', color='black'),
                          flierprops=dict(markerfacecolor='green', marker='D'))

        plt.title('Box Plot of Accuracy Over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.show()