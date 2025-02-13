import torch
from torch.utils.data import Dataset, DataLoader, random_split
import gp
import string

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=10):
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
full_dataset = TextDataset("../data/alice.txt", seq_length=20)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Create DataLoaders. Need drop_last true to ensure hidden state at t-1 will be compatible with input
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

'''Individual Tests'''
# interpreter = Interpreter(train=train_loader, test=test_loader, activation="relu", auto_bias=True, embedding=True, embed_dim=128, vocab_size=70)
# instructions = Instructions(activation="relu")
# genome = gp.Genome(train=train_loader, test=test_loader, interpreter=interpreter, instructions=instructions)
# genome.genome = [
#     'matmul', 'matmul','matmul','matmul','matmul','matmul','matmul','matmul','matmul','matmul', 125, 250, 500, 1000, 500, 250, 125, 64
#     # 'mat_add', 'mat_add', 4, 16, 'mat_add_nodes', 'mat_add', 3, 256, 5, 2, 128, 5, 'mat_add_nodes', 32, 'matmul_nodes',
#     #  4, 32, 'mat_add', 'mat_add_nodes', 'maxpool2d', 'maxpool2d', 4, 8, 5, 'maxpool2d'
# ]
# network = genome.transcribe()
# print(network)
# network.fit(epochs=5)
# fitness = network.evaluate()
# print(f"Genome fitness: {fitness}")
#
# # Model Inference
# text = "Alice went to wonderland and saw a"
# for _ in range(100):
#     input_seq = torch.tensor([full_dataset.char2idx[ch] for ch in text[-20:]], dtype=torch.long).unsqueeze(0)
#     pred = network.forward(input_seq)
#     pred_last = pred[:, -1, :]
#     probs = F.softmax(pred_last, dim=-1)
#     probs = probs.squeeze(0)
#     next_char_id = torch.multinomial(probs, num_samples=1).item()
#     next_char = full_dataset.idx2char[next_char_id]
#     text += next_char
#
# print(text)

'''Population Tests'''
# pop = Population.load("pop.pkl")
pop = gp.Population(
    size=50, # Population size (number of individuals)
    num_initial_genes=50, # Number of genes to start with for each individual
    input_shape=(20,), # Input shape
    output_shape=(20,128), # Output shape
    activation="relu", # Activation function to use (of None, no default activation function is used)
    auto_bias=True, # Whether to automatically add bias to the network
    separate_ints=True, # Whether to separate small integers from large integers in the stacks
    embedding=True,
    embed_dim=128,
    vocab_size=70,
    recurrent=True
)
# pop.save("pop.pkl")
pop.run(
    train=train_loader,
    test=test_loader,
    generations=20, # Number of generations to run this population for
    epochs=1, # Number of epochs to train each network for
    method='tournament', # Selection method
    pool_size=15, # Number of individuals to select from the population for each selection into the next generation
    param_limit=50000000, # Maximum number of parameters allowed in a network
    flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
    drought=False, # Whether to use a drought mechanism that kills bad networks off early
    increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
    downsample=0.1 # Choose whether to downsample and by how much
)

for genome in pop.population:
    print(genome.fitness)
    print(genome.genome)
    print("")