from utils import median_absolute_deviation
from interpreter import Interpreter
from instructions import Instructions
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import copy
import pickle

# TODO: Choose mult values based on the input size. Basically just multiples of the input going in either direction. Good for speed, reduces amount of weird numbers
INT_RANGE = (1, 5)
MULT_VALS = [4, 8, 16, 32, 64, 128, 256]
ADD_RATE = 0.18
REMOVE_RATE = ADD_RATE/(1 + ADD_RATE)
# TODO: Experiment with alpha
ALPHA = 0.5 # Used for loss function with parameter count. Between 0 and 1

class Genome:
    '''Genome of a Push Program'''
    def __init__(self, train, test, interpreter, instructions):
        self.genome = []
        self.fitness = 0
        self.metrics = (float('inf'), float('-inf'), float('inf')) # Loss, accuracy, parameter count
        self.train = train
        self.test = test
        self.interpreter = interpreter
        self.instructions = instructions
        self.network = None
        self.results = {} # Results of the network on the test data

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
        # TODO: Assign probabilities of adding each type. Ideally prevent the tiny matmuls
        data_type = random.randint(0, 1)

        match data_type:
            case 0:
                int_type = random.randint(0, 1)
                match int_type:
                    case 0:
                        return random.choice(MULT_VALS) # Random multiple of input size
                    case 1:
                        return random.randint(*INT_RANGE) # Random integer
            case 1:
                return random.choice(self.instructions.instructions)  # Add instruction. Project to list for random.choice to work

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
    # TODO: Move interpreter instance to population
    def transcribe(self):
        '''Transcribes the genome to create a network. Returns the network'''
        self.interpreter.read_genome(self.genome) # Read genome (process it into stacks)
        self.network = self.interpreter.run() # Generate network object
        self.interpreter.clear() # Clear stacks
        return self.network

class Population:
    '''Population of Push Program Genomes'''
    def __init__(self, size, num_initial_genes, train, test, activation, auto_bias):
        self.size = size
        self.train = train
        self.test = test
        self.instructions = Instructions(activation=activation)
        self.interpreter = Interpreter(train=self.train, test=self.test, activation=activation, auto_bias=auto_bias) # Pass shapes to interpreter
        self.population = [Genome(train, test, self.interpreter, self.instructions) for _ in range(size)]
        # Initialize the population with random genes
        for genome in self.population:
            genome.initialize_random(num_initial_genes)

    def save(self, filename):
        '''Saves the population to a file.'''
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Population saved to {filename}")

    @staticmethod
    def load(filename):
        '''Loads a population from a file.'''
        with open(filename, 'rb') as file:
            population = pickle.load(file)
        print(f"Population loaded from {filename}")
        return population

    def tournament(self, size):
        '''Selects the best genome from a tournament with size individuals'''
        size = min(size, self.size) # Ensure size is not greater than the population size
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
        self.population.sort(key=lambda x: x.fitness[1], reverse=True) # Sort by accuracy currently
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

        self.save('pop.pkl')

        # Mutate the new population
        for genome in self.population:
            genome.UMAD()

    def run(self, generations, epochs, method='tournament', pool_size=5, param_limit=500000):
        '''Runs the population on the train and test data'''
        acc = []
        size = []
        best_genome = (None, float('-inf'))
        for gen_num in range(1, generations + 1):
            gen_acc = [] # Store accuracies for graphing
            gen_size = [] # Store sizes for graphing
            param_max = 0
            for genome in self.population:
                gen_size.append(len(genome.genome))
                network = genome.transcribe()
                if network.param_count < param_limit: # If network is small enough
                    print(network)
                    # Train the network
                    network.fit(epochs=epochs)

                    # Evaluate the network, store results
                    loss, accuracy, results = network.evaluate()
                    param_max = max(param_max, network.param_count)
                    genome.metrics = (loss, accuracy, network.param_count)
                    genome.results = results

                    # Update best genome
                    if accuracy > best_genome[1]:
                        best_genome = (genome, accuracy)

                    gen_acc.append(accuracy)
                    print(f"Genome Accuracy: {accuracy}")
                    print(f"Genome Loss: {loss}")
                else:
                    genome.fitness = (float('inf'), 0) # Worst possible fitness
                    gen_acc.append(0)
                    for batch in range(len(self.test)):
                        genome.results[batch] = (float('inf'), float('-inf')) # Worst possible case on each batch

                # Prevent memory leaks by clearing cuda cache
                del network
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update fitness using normalized param count
            # TODO: Incorporate accuracy and loss into once fitness function somehow?
            for genome in self.population:
                p_norm = genome.metrics[2] / param_max # Just parameter count / max parameter count
                genome.fitness = (
                    # TODO: Normalization may not be good for loss. Probably too small
                    (1 - ALPHA) * genome.metrics[1] + ALPHA * p_norm, # Loss
                    (1 - ALPHA) * genome.metrics[0] - ALPHA * p_norm, # Accuracy
                )

            acc.append(gen_acc)
            size.append(gen_size)

            print("\n--------------------------------------------------")
            print(f"Generation {gen_num} Completed")
            print("--------------------------------------------------\n")

            self.forward_generation(method=method, size=pool_size)

        if best_genome[0] is not None:
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

        # Save accuracy data to csv file for excel plotting
        acc_csv = np.array(acc)
        np.savetxt("accuracy.csv", acc_csv, delimiter=",")