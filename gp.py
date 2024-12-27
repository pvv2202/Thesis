from utils import median_absolute_deviation
from interpreter import Interpreter
from instructions import Instructions
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import copy

INT_RANGE = (1, 256)
FLOAT_RANGE = (0.0, 1.0)
ADD_RATE = 0.18
REMOVE_RATE = ADD_RATE/(1 + ADD_RATE)

# TODO: Take random genomes and run UMAD a bunch just to see what happens. Neutral landscapes?
# TODO: Have int equal probability of adding small ints and a larger range?

class Genome:
    '''Genome of a Push Program'''
    def __init__(self, train, test, activation):
        self.genome = []
        self.fitness = 0
        self.train = train
        self.test = test
        self.activation = activation
        self.network = None
        self.results = None

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
        self.train = train
        self.test = test
        self.population = [Genome(train, test, activation) for _ in range(size)]
        # Initialize the population with random genes
        for genome in self.population:
            genome.initialize_random(num_initial_genes)

    def tournament(self, size):
        '''Selects the best genome from a tournament with size individuals'''
        tournament = random.sample(self.population, size)
        return max(tournament, key=lambda x: x.fitness)

    def epsilon_lexicase(self, candidates, round, max_rounds):
        '''Selects the best genome using epsilon lexicase selection'''
        # Choose a random test case
        batch = random.randint(0, len(self.test) - 1)

        # Randomly select whether to use loss (0) or accuracy (1)
        # metric = random.choice([0, 1]) # TODO: Temporarily just using accuracy
        metric = 1

        # Get the results for the test case
        test_results = [genome.results[batch][metric] for genome in candidates] # Results of the form total loss, percent accuracy

        # Calculate the median and median absolute deviation (MAD)
        median = np.median(test_results)
        mad = median_absolute_deviation(test_results)

        # Define epsilon as a range around the median
        epsilon = 2 * mad
        lower_bound = median - epsilon
        upper_bound = median + epsilon

        # Get the fitness of each genome on the test case
        next = [genome for genome in candidates if lower_bound <= genome.results[batch][metric] <= upper_bound]

        # If no genomes pass, fallback to the best genome based on the test case
        if not next:
            if metric == 0:
                return min(self.population, key=lambda genome: genome.results[batch][metric])
            if metric == 1:
                return max(self.population, key=lambda genome: genome.results[batch][metric])

        if len(next) == 1:
            return next[0]
        else:
            # Randomly select from passing genomes unless we're on the max rounds at which point just return a random one
            return self.epsilon_lexicase(next, round + 1, max_rounds) if round < max_rounds else random.choice(next)

    def forward_generation(self, method='tournament', size=5, max_rounds=5):
        '''Moves the population forward one generation'''
        # Sort the population by fitness. Higher fitness is better
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        print([genome.fitness[1] for genome in self.population]) # Should print accuracy

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
            case 'epsilon_lexicase':
                new_population = []
                for _ in range(self.size):
                    # Select a genome and make a deep copy. We pass results and a random sample of the population
                    genome = self.epsilon_lexicase(self.population, 1, max_rounds)
                    new_genome = copy.deepcopy(genome)
                    new_population.append(new_genome)
                # Update the population
                self.population = new_population

        # Mutate the new population
        for genome in self.population:
            genome.UMAD()

    def run(self, generations, epochs, method='tournament', pool_size=5):
        '''Runs the population on the train and test data'''
        acc = []
        size = []
        best_genome = (None, float('-inf'))
        for gen_num in range(1, generations + 1):
            gen_acc = []
            gen_size = []
            gen_results = {}
            for genome in self.population:
                gen_size.append(len(genome.genome))
                network = genome.transcribe()
                print(network)
                # Train the network
                network.fit(epochs=epochs)

                # Evaluate the network
                loss, accuracy, results = network.evaluate()
                genome.fitness = (loss, accuracy)
                genome.results = results

                # Update best genome
                if accuracy > best_genome[1]:
                    best_genome = (genome, accuracy)

                gen_acc.append(accuracy)
                print(f"Genome Accuracy: {accuracy}")
                print(f"Genome Loss: {loss}")
            acc.append(gen_acc)
            size.append(gen_size)

            print("\n--------------------------------------------------")
            print(f"Generation {gen_num} Completed")
            print("--------------------------------------------------\n")

            self.forward_generation(method=method, size=pool_size)

        print(f"Best genome: {best_genome[0].genome}")

        # Generate labels for each generation
        labels = [i for i in range(1, generations + 1)]

        # Create box plot for size
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed
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
        plt.tight_layout()  # Automatically adjusts layout to fit elements
        plt.show()

        # Create box plot for accuracy
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed
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
        plt.tight_layout()  # Automatically adjusts layout to fit elements
        plt.show()
