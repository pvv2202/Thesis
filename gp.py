from keras.src.metrics.accuracy_metrics import accuracy

from interpreter import Interpreter
from instructions import Instructions
import matplotlib.pyplot as plt
import random
import torch
import copy

INT_RANGE = (1, 256)
FLOAT_RANGE = (0.0, 1.0)
ADD_RATE = 0.18
REMOVE_RATE = ADD_RATE/(1 + ADD_RATE)

# TODO: Take random genomes and run UMAD a bunch just to see what happens. Neutral landscapes?

class Genome:
    '''Genome of a Push Program'''
    def __init__(self, train, test, activation):
        self.genome = []
        self.fitness = 0
        self.train = train
        self.test = test
        self.activation = activation
        self.network = None

        # Initialize instructions
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
        type = random.randint(0, 1)

        match type:
            case 0:
                return random.randint(*INT_RANGE) # Random integer
            case 1:
                return random.choice(self.instructions.instructions)  # Add instruction. Project to list for random.choice to work
            case 2:
                return random.uniform(*FLOAT_RANGE) # Random float

    def UMAD(self):
        '''
        Mutates the genome using UMAD. With some add probability, add a gene before or after each gene. Loop
        through genome again. With remove probability = add probability/(1 + add probability), remove a gene
        '''
        # Add genes
        add_genome = []
        for gene in self.genome:
            if random.random() <= ADD_RATE:
                if random.random() < 0.5:
                    # Add before
                    add_genome.append(gene)
                    add_genome.append(self.random_gene())
                else:
                    # Add after
                    add_genome.append(self.random_gene())
                    add_genome.append(gene)
            else:
                add_genome.append(gene)

        # Remove genes
        new_genome = []
        for gene in add_genome:
            # If it's going to be removed, just don't add it
            if random.random() <= REMOVE_RATE:
                continue
            new_genome.append(gene)

        # Update genome
        self.genome = new_genome

    def transcribe(self):
        '''Transcribes the genome to create a network. Returns the network'''
        interpreter = Interpreter(self.train, self.test, self.activation) # Pass shapes to interpreter
        interpreter.read_genome(self.genome)
        self.network = interpreter.run()
        return self.network

class Population:
    '''Population of Push Program Genomes'''
    def __init__(self, size, num_initial_genes, train, test, activation=torch.softmax):
        self.size = size
        self.population = [Genome(train, test, activation) for _ in range(size)]
        # Initialize the population with random genes
        for genome in self.population:
            genome.initialize_random(num_initial_genes)

    def tournament(self, size):
        '''Selects the best genome from a tournament with size individuals'''
        tournament = random.sample(self.population, size)
        return max(tournament, key=lambda x: x.fitness)

    def forward_generation(self, method='tournament', size=5):
        '''Moves the population forward one generation'''
        # Sort the population by fitness. Higher fitness is better
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        print([genome.fitness for genome in self.population])

        match method:
            case 'tournament':
                new_population = []
                for _ in range(self.size):
                    # Select a genome and make a deep copy
                    genome = self.tournament(size)
                    new_genome = copy.deepcopy(genome)
                    new_population.append(new_genome)
                # Update the population
                self.population = new_population
            case 'elite':
                new_population = []
                for i in range(size):
                    # Make deep copies of the top genomes
                    new_genome = copy.deepcopy(self.population[i])
                    new_population.append(new_genome)
                # Fill the rest of the population with mutated copies
                while len(new_population) < self.size:
                    genome = random.choice(new_population)
                    new_genome = copy.deepcopy(genome)
                    new_genome.UMAD()
                    new_population.append(new_genome)
                self.population = new_population

        # Mutate the new population
        for genome in self.population:
            genome.UMAD()

    def run(self, generations, epochs):
        '''Runs the population on the train and test data'''
        acc = []
        size = []
        best_genome = (None, float('-inf'))
        for gen_num in range(1, generations + 1):
            gen_acc = []
            gen_size = []
            for genome in self.population:
                gen_size.append(len(genome.genome))
                network = genome.transcribe()
                print(network)
                # Train the network
                network.fit(epochs=epochs)

                # Evaluate the network
                loss, accuracy = network.evaluate()
                genome.fitness = accuracy

                # Update best genome
                if accuracy > best_genome[1]:
                    best_genome = (genome, accuracy)

                gen_acc.append(genome.fitness)
                print(f"Genome fitness: {genome.fitness}")
            acc.append(gen_acc)
            size.append(gen_size)

            print("\n--------------------------------------------------")
            print(f"Generation {gen_num} Completed")
            print("--------------------------------------------------\n")

            self.forward_generation(method='tournament', size=5)

        print(f"Best genome: {best_genome[0].genome}")

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