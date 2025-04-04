from utils import median_absolute_deviation
from interpreter import Interpreter
from instructions import Instructions
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import copy
import pickle
import heapq
import time

# TODO: Choose mult values based on the input size. Basically just multiples of the input going in either direction. Good for speed, reduces amount of weird numbers
SINT_RANGE = (1, 5)
INT_VALS = [16, 32, 64, 128, 256]
ADD_RATE = 0.08
REMOVE_RATE = ADD_RATE/(1 + ADD_RATE)
# TODO: Experiment with alpha
ALPHA = 0.2 # Used for loss function with parameter count. Between 0 and 1. Higher means we weigh parameter count more

class Genome:
    """Genome of a Push Program"""
    def __init__(self, interpreter, instructions, mute_instructions=None):
        self.genome = []
        self.fitness = 0
        self.metrics = (float('inf'), float('-inf'), float('inf')) # Loss, accuracy, parameter count
        self.interpreter = interpreter
        self.instructions = instructions
        if mute_instructions is not None:
            self.valid_instructions = [i for i in self.instructions.instructions if i not in mute_instructions]
        else:
            self.valid_instructions = self.instructions.instructions

        self.network = None
        self.results = [] # Results of the network on the test data

    def random_index(self):
        """Returns a random index in the genome"""
        return random.randint(0, len(self.genome))

    def initialize_random(self, num_genes):
        """Initializes the genome with random genes"""
        # Add genes
        for _ in range(num_genes):
            self.genome.append(self.random_gene())

    def random_gene(self):
        """Returns a random gene"""
        # Randomly select what type of thing to add
        data_type = random.random()

        if data_type < 0.4:
            int_type = random.randint(0, 1)
            match int_type:
                case 0:
                    return random.choice(INT_VALS)  # Random multiple of input size
                case 1:
                    return random.randint(*SINT_RANGE)  # Random integer
        elif data_type < 0.8:
            return random.choice(self.valid_instructions)  # Add instruction. Project to list for random.choice to work

        else:
            return '('

    def umad(self, gen_num, count=0):
        """
        Mutates the genome using UMAD. With some add probability, add a gene before or after each gene. Loop
        through genome again. With remove probability = add probability/(1 + add probability), remove a gene
        """
        # Add genes
        # TODO: Maybe I want to have add on left/right equal chance instead of both in one add case? Try graphing best performers vs. size?
        if gen_num <= -1:
            self.genome.insert(0, self.random_gene())
        else:
            add_genome = []
            for gene in self.genome:
                if random.random() <= ADD_RATE:
                    if random.random() < 0.5:
                        # Add before
                        add_genome.append(self.random_gene())
                        add_genome.append(gene)
                    else:
                        # Add after
                        add_genome.append(gene)
                        add_genome.append(self.random_gene())
                else:
                    add_genome.append(gene)

            # Handle special case where genome is empty
            if len(self.genome) == 0:
                if random.random() <= ADD_RATE:
                    add_genome.append(self.random_gene())

            # Remove genes
            new_genome = []
            for gene in add_genome:
                # If it's going to be removed, just don't add it
                if random.random() <= REMOVE_RATE:
                    continue
                new_genome.append(gene)

            # Redo if no changes were made. Only do this 100 times
            if self.genome == new_genome and count < 100:
                self.umad(gen_num, count=count+1)

            # Update genome
            self.genome = new_genome

    def transcribe(self):
        """Transcribes the genome to create a network. Returns the network"""
        self.interpreter.read_genome(self.genome) # Read genome (process it into stacks)
        self.network = self.interpreter.run() # Generate network object
        self.interpreter.clear() # Clear stacks
        return self.network

class Population:
    """Population of Push Program Genomes"""
    def __init__(self, size, num_initial_genes, input_shape, output_shape, activation, auto_bias,
                 separate_ints, mute_instructions=None, embedding=False, embed_dim=None, recurrent=False,
                 out_file=None):
        self.size = size
        self.instructions = Instructions(activation=activation)
        # Interpreter needs the data (for auto output) and a number of other toggleable options
        self.interpreter = Interpreter(
            input_shape=input_shape,
            output_shape=output_shape,
            instructions=self.instructions,
            activation=activation,
            auto_bias=auto_bias,
            separate_ints=separate_ints,
            embedding=embedding,
            embed_dim=embed_dim,
            recurrent=recurrent
        )
        self.population = [Genome(self.interpreter, self.instructions, mute_instructions) for _ in range(size)]
        self.out_file = out_file
        # Initialize the population with random genes
        for genome in self.population:
            genome.initialize_random(random.randint(*num_initial_genes))

    def save(self, filename):
        """Saves the population to a file."""
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Population saved to {filename}")

    @staticmethod
    def load(filename):
        """Loads a population from a file."""
        with open(filename, 'rb') as file:
            population = pickle.load(file)
        print(f"Population loaded from {filename}")
        return population

    def tournament(self, size, minimize=False):
        """Selects the best genome from a tournament with size individuals"""
        size = min(size, self.size) # Ensure size is not greater than the population size
        tournament = random.sample(self.population, size)

        if minimize:
            return min(tournament, key=lambda x: x.fitness[0])
        else:
            return max(tournament, key=lambda x: x.fitness[0]) # By accuracy

    def epsilon_lexicase(self, candidates, test, metric_index=1, minimize=False):
        """Selects the best genome using epsilon-lexicase selection"""
        test_order = [i for i in range(len(test))] # Indices. Test batches aren't shuffled so indices work
        random.shuffle(test_order)

        for t in test_order:
            metric = [genome.results[t][metric_index] for genome in candidates]

            if minimize:
                best = min(metric)
            else:
                best = max(metric)

            mad = median_absolute_deviation(metric)
            eps = 1 * mad # 1 * mad for now. Something to test

            if minimize:
                threshold = best + eps
            else:
                threshold = best - eps

            survivors = []
            for g, m in zip(candidates, metric):
                if minimize:
                    if m <= threshold:
                        survivors.append(g)
                else:
                    if m >= threshold:
                        survivors.append(g)

            candidates = survivors

            if len(candidates) == 1:
                return candidates[0]

            if not candidates:
                if minimize:
                    return min(candidates, key=lambda x: x.results[t][metric_index])
                else:
                    return max(candidates, key=lambda x: x.results[t][metric_index])

        # End cases
        if len(candidates) > 1:
            return random.choice(candidates) # Randomly select from passing genomes
        elif len(candidates) == 1:
            return candidates[0] # Select the best
        else:
            return random.choice(candidates) # Randomly select from the population if nothing passed (shouldn't happen)

    def forward_generation(self, test, gen_num, method='tournament', size=5):
        """Moves the population forward one generation"""
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
                    #  genome.metrics = (loss, accuracy, network.flops, network.param_count)
                    genome = self.epsilon_lexicase(self.population, test, metric_index=1, minimize=False)
                    new_genome = copy.deepcopy(genome)
                    new_population.append(new_genome)
                # Update the population
                self.population = new_population

        self.save('pop.pkl')

        # Mutate the new population
        for genome in self.population:
            genome.umad(gen_num, count=0)

    def run(self, train, test, generations, epochs, loss_fn, optimizer=torch.optim.Adam, method='tournament',
            pool_size=5, param_limit=1000000, flops_limit=50000000, increase_epochs=False):
        """Runs the population on the train and test data"""
        start = time.time()
        acc = []
        l = []
        size = []
        best_genomes_acc = []
        heapq.heapify(best_genomes_acc)
        best_genomes_loss = []
        heapq.heapify(best_genomes_loss)
        counter = 0
        for gen_num in range(1, generations + 1):
            gen_acc = [] # Store accuracies for graphing
            gen_size = [] # Store sizes for graphing
            gen_loss = [] # Store losses for graphing
            param_max = 0
            flops_max = 0
            for genome in self.population:
                print(genome.genome)
                gen_size.append(len(genome.genome))
                network = genome.transcribe()

                # Delete random genes until the parameter count is below the limit
                while network.param_count > param_limit:
                    i = random.randint(0, max(0, len(genome.genome) - 1))
                    del genome.genome[i]
                    network = genome.transcribe()

                # Delete random genes until the flops count is below the limit
                while network.flops > flops_limit:
                    i = random.randint(0, max(0, len(genome.genome) - 1))
                    del genome.genome[i]
                    network = genome.transcribe()

                print(network)

                # Determine value to pass for generation
                if increase_epochs:
                    generation = gen_num/generations # Pass the generation number as a fraction
                else:
                    generation = None

                # Train the network
                network.fit(train=train, epochs=epochs, loss_fn=loss_fn, optimizer=optimizer, generation=generation)
                loss, accuracy, results = network.evaluate(test=test, loss_fn=loss_fn)
                param_max = max(param_max, network.param_count)
                flops_max = max(flops_max, network.flops)
                genome.metrics = (loss, accuracy, network.flops, network.param_count)
                genome.results = results

                # Update best genomes. Include counter to break ties
                if len(best_genomes_acc) < 100:
                    heapq.heappush(best_genomes_acc, (accuracy, counter, copy.deepcopy(genome.genome)))
                else:
                    heapq.heappushpop(best_genomes_acc, (accuracy, counter, copy.deepcopy(genome.genome)))

                if len(best_genomes_loss) < 100:
                    heapq.heappush(best_genomes_loss, (-loss, counter, copy.deepcopy(genome.genome)))
                else:
                    heapq.heappushpop(best_genomes_loss, (-loss, counter, copy.deepcopy(genome.genome)))

                counter += 1

                gen_acc.append(accuracy)
                gen_loss.append(loss)
                print(f"Genome Accuracy: {accuracy}")
                print(f"Genome Loss: {loss}")

                # Prevent memory leaks by clearing cuda cache
                del network
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update fitness using normalized flops. Don't actually need this anymore
            for genome in self.population:
                genome.fitness = (
                    genome.metrics[1],
                    genome.metrics[0]
                )

            acc.append(gen_acc)
            l.append(gen_loss)
            size.append(gen_size)

            # TODO: Do this for loss and size as well
            # Save accuracy procedurally for graphing
            if self.out_file is not None:
                acc_csv = np.array(acc)
                np.savetxt(self.out_file.csv, acc_csv, delimiter=",")

            print("\n--------------------------------------------------")
            print(f"Generation {gen_num} Completed")
            print("--------------------------------------------------\n")

            self.forward_generation(test, gen_num, method=method, size=pool_size)

        print(f"EXECUTION TIME: {time.time() - start} seconds")

        print("BEST BY ACCURACY")
        for genome in best_genomes_acc:
            print(genome[0])
            print(genome[2])

        print("BEST BY LOSS")
        for genome in best_genomes_loss:
            print(-genome[0])
            print(genome[2])

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

        # Create box plot for loss
        plt.figure(figsize=(10, 6))  # Adjust width and height as needed
        size_plot = plt.boxplot(l,
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
        if self.out_file is not None:
            acc_csv = np.array(acc)
            np.savetxt(f"{self.out_file}_acc.csv", acc_csv, delimiter=",")

        # Save loss data to csv file for excel plotting
        if self.out_file is not None:
            acc_csv = np.array(l)
            np.savetxt(f"{self.out_file}_loss.csv", acc_csv, delimiter=",")

        # Save size data to csv file for excel plotting
        if self.out_file is not None:
            acc_csv = np.array(size)
            np.savetxt(f"{self.out_file}_size.csv", acc_csv, delimiter=",")