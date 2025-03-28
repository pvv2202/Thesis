import torch
from torch.utils.data import Dataset, DataLoader, random_split
import gp
import string

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length):
        with open(file_path, "r", encoding="utf-8") as file:
            self.text = file.read().lower()

        # Define fixed character set
        self.chars = string.ascii_lowercase + string.digits + string.punctuation + " \n"
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Remove unseen characters and encode text
        self.encoded_text = [self.char2idx[ch] for ch in self.text if ch in self.char2idx]
        self.seq_length = seq_length

    def __len__(self):
        return len(self.encoded_text) - self.seq_length

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.encoded_text[idx:idx+self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.encoded_text[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return input_seq, target_seq

# Load dataset
full_dataset = TextDataset(file_path="data/alice.txt", seq_length=20)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders. Need drop_last true to ensure hidden state at t-1 will be compatible with input
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

'''Individual Tests'''
# instructions = gp.Instructions(activation=None)
# interpreter = gp.Interpreter(input_shape=(1,), output_shape=(128,), instructions=instructions, activation=None, auto_bias=False, embedding=torch.nn.Embedding(num_embeddings=70, embedding_dim=128), embed_dim=128, recurrent=True)
# genome = gp.Genome(interpreter=interpreter, instructions=instructions)
# genome.genome = [
#     'await_connection', '(', 3, 'maxpool2d', '(', 'matmul_nodes', '(', '(', 'mat_add', 'dup', 'avgpool2d',
#      'mat_add_nodes', 'maxpool2d', 'identity', 5, 'for_n', 'back_connect', 'matmul', '(', 'conv2d', '(', 'mat_add', '(',
#      128, '(', 'back_connect', 'matmul', 3, 'back_connect', '(', 5, 3, 'identity', '(', 'conv2d', 'mat_add_nodes', '(',
#      16, 'await_connection', 16
#     # 'mat_add', 'mat_add', 4, 16, 'mat_add_nodes', 'mat_add', 3, 256, 5, 2, 128, 5, 'mat_add_nodes', 32, 'matmul_nodes',
#     #  4, 32, 'mat_add', 'mat_add_nodes', 'maxpool2d', 'maxpool2d', 4, 8, 5, 'maxpool2d'
# ]
# network = genome.transcribe()
# print(network)
# network.fit(epochs=5, train=train_loader)
# fitness = network.evaluate(test=test_loader)
# print(f"Genome fitness: {fitness}")


'''Population Tests'''
# pop = Population.load("pop.pkl")
pop = gp.Population(
    size=50, # Population size (number of individuals)
    num_initial_genes=(5, 100), # Number of genes to start with for each individual
    input_shape=(1,), # Input shape
    output_shape=(128,), # Output shape
    activation="relu", # Activation function to use (of None, no default activation function is used)
    auto_bias=True, # Whether to automatically add bias to the network
    separate_ints=True, # Whether to separate small integers from large integers in the stacks
    mute_instructions=['flatten', 'transpose', 'layer_norm', 'batch_norm'], # Instructions to mute
    embedding=torch.nn.Embedding(num_embeddings=70, embedding_dim=128),
    embed_dim=128,
    recurrent=True
)
# pop.save("pop.pkl")
pop.run(
    train=train_loader,
    test=test_loader,
    generations=20, # Number of generations to run this population for
    epochs=1, # Number of epochs to train each network for
    loss_fn=torch.nn.CrossEntropyLoss(), # Loss function
    optimizer=torch.optim.Adam,
    method='epsilon-lexicase', # Selection method
    pool_size=15, # Number of individuals to select from the population for each selection into the next generation
    param_limit=50000000, # Maximum number of parameters allowed in a network
    flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
    increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
    downsample=0.1 # Choose whether to downsample and by how much
)

for genome in pop.population:
    print(genome.fitness)
    print(genome.genome)
    print("")