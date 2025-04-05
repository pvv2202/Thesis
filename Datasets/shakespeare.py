import torch
from torch.utils.data import Dataset, DataLoader, random_split
import gp
import csv

class Shakespeare(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, "r") as file:
            csv_reader = csv.reader(file)
            self.texts = []
            curr_text = ""
            prev_title = None

            for row in csv_reader:
                if prev_title is None:
                    prev_title = row[1]
                elif row[1] != prev_title:
                    self.texts.append(curr_text)
                    curr_text = ""
                    prev_title = row[1]

                curr_text += row[5]

        # Define character set
        self.vocab = sorted(list(set("".join(self.texts))))
        self.char_to_i = {char: i for i, char in enumerate(self.vocab)}
        self.i_to_char = {i: char for char, i in self.char_to_i.items()}

        self.encoded_texts = [[self.char_to_i[char] for char in text] for text in self.texts]
        self.seq_length = seq_length

        # Create list of indices for everything to allow for sampling
        self.indices = []
        for i, text in enumerate(self.encoded_texts):
            for j in range(len(text) - seq_length):
                self.indices.append((i, j))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i, j = self.indices[idx]
        text = self.encoded_texts[i]
        input_seq = torch.tensor(text[j : j + self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(text[j + 1 : j + self.seq_length + 1], dtype=torch.long)
        return input_seq, target_seq

# Load dataset
dataset = Shakespeare(file_path="data/shakespeare.csv", seq_length=20)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

instructions = gp.Instructions(activation=None)
interpreter = gp.Interpreter(input_shape=(1,), output_shape=(len(dataset.vocab),), instructions=instructions, activation=None, auto_bias=False, embedding=torch.nn.Embedding(num_embeddings=len(dataset.vocab), embedding_dim=50), embed_dim=50, recurrent=True)
genome = gp.Genome(interpreter=interpreter, instructions=instructions)
genome.genome = [
    256, 'sigmoid', '(', 'maxpool2d', 4, '(', '(', 'mat_add', 4, 'mat_add', 2, '(', '(', '(', 256, 1, 'matmul', '(', 'maxpool2d', '(', 'mat_add_nodes', '(', 'matmul', 'conv2d', '(', 'dup', 'maxpool2d', 'identity', 2, 'back_connect', '(', '(', '(', '(', 64, 4, 'matmul', 16, 'await_connection', 2, 3, 'tanh', 'await_connection', 'matmul', 1, 'conv2d', 'back_connect', 2
]
network = genome.transcribe()
print(network)
network.fit(epochs=5, train=train_loader)
fitness = network.evaluate(test=test_loader)
print(f"Genome fitness: {fitness}")

# '''Population Tests'''
# # pop = Population.load("pop.pkl")
# pop = gp.Population(
#     size=50, # Population size (number of individuals)
#     num_initial_genes=(5, 100), # Number of genes to start with for each individual
#     input_shape=(1,), # Input shape
#     output_shape=(len(dataset.vocab),), # Output shape
#     activation=None, # Activation function to use (of None, no default activation function is used)
#     auto_bias=False, # Whether to automatically add bias to the network
#     separate_ints=True, # Whether to separate small integers from large integers in the stacks
#     mute_instructions=[], # Instructions to mute
#     embedding=torch.nn.Embedding(num_embeddings=len(dataset.vocab), embedding_dim=50),
#     embed_dim=50,
#     recurrent=True
# )
# # pop.save("pop.pkl")
# pop.run(
#     train=train_loader,
#     test=test_loader,
#     generations=20, # Number of generations to run this population for
#     epochs=1, # Number of epochs to train each network for
#     loss_fn=torch.nn.CrossEntropyLoss(), # Loss function
#     optimizer=torch.optim.Adam,
#     method='epsilon-lexicase', # Selection method
#     pool_size=15, # Number of individuals to select from the population for each selection into the next generation
#     param_limit=50000000, # Maximum number of parameters allowed in a network
#     flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
#     increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
# )
#
# for genome in pop.population:
#     print(genome.fitness)
#     print(genome.genome)
#     print("")