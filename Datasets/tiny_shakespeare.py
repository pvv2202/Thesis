from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import random
import string
import gp
import torch

class StridedTextDataset(Dataset):
    def __init__(self, file_path, seq_length, stride=1):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read().lower()

        # Define a fixed character set
        self.chars = string.ascii_lowercase + string.digits + string.punctuation + " \n"
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

        # Encode text
        self.encoded_text = [self.char2idx[ch] for ch in text if ch in self.char2idx]
        self.seq_length = seq_length

        # Sample positions based on stride
        self.indices = list(range(0, len(self.encoded_text) - seq_length - 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        input_seq = torch.tensor(self.encoded_text[i:i+self.seq_length], dtype=torch.long)
        target_seq = torch.tensor(self.encoded_text[i+1:i+self.seq_length+1], dtype=torch.long)
        return input_seq, target_seq

# Load dataset
dataset = StridedTextDataset(file_path="data/tiny_shakespeare.txt", seq_length=50, stride=16)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders. Need drop_last true to ensure hidden state at t-1 will be compatible with input
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

'''Individual Tests'''
# instructions = gp.Instructions(activation=None)
# interpreter = gp.Interpreter(input_shape=(1,), output_shape=(len(dataset.chars),), instructions=instructions, activation=None, auto_bias=False, embedding=torch.nn.Embedding(num_embeddings=len(dataset.chars), embedding_dim=50), embed_dim=50, recurrent=True)
# # genome = gp.Genome(interpreter=interpreter, instructions=instructions)
# genome.genome = [
# # '(', 256, 'await_connection', 'relu', 'dup', 'layer_norm', 32, 64, 4, '(', 5, 256, 5, 256, '(', 'tanh', 'flatten', '(', 'transpose', 'layer_norm', 'tanh', 'matmul_nodes', 32, 1, '(', 'layer_norm', '(', '(', '(', 32, 'mat_add', 'identity', 32, '(', 4, 64, '(', '(', '(', 128, 32, 'await_connection', '(', 128, 256, '(', 'matmul', 3, 16, 3, '(', 32, 'flatten', 'flatten', 'relu', 'relu', '(', 32, 32, 1, 'back_connect', 'mat_add_nodes', 64, 'identity', 32, 'mat_add_nodes', 'relu', 5, '(', 'dup', 32, 'mat_add', 'matmul', 5, 64, 'flatten', 3, 'tanh', 'back_connect', 256, 256, 128, 'flatten', 4, 'back_connect', 'flatten', 'matmul_nodes', 16, '(', 'dup', 'transpose', 16, 'transpose', 'relu', 'sigmoid', 'mat_add_nodes', 16, '(', 3, 'back_connect', 128, 'flatten', 5, '(', 'mat_add_nodes', 'relu', 'dup', 'await_connection', 128, 3, 3, '('
#
# # 32, 2, 4, 'mat_add_nodes', 'layer_norm', 4, 4, 5, 1, 'transpose', '(', 'tanh', '(', 16, 'matmul_nodes', 2, 'layer_norm', 32, '(', 'matmul', 'await_connection', 'layer_norm', 32, 'layer_norm', 2, 'relu', '(', 'tanh', 5, 'transpose', 'transpose', 'mat_add_nodes', '(', 32, 'sigmoid', 2, 'mat_add_nodes', 'relu', 'transpose', 64, 'identity', 32, 256, 'back_connect', 5, 5, 'identity', '(', 32, 64, 'flatten', 'mat_add', 'back_connect', 1, 256, 'transpose', 16, 'await_connection', '(', 16, 'transpose', 256, 'identity', 128, 'layer_norm', '(', 128, 64, 256, 4, '(', 'transpose'
#
# # (', 256, 2, 32, 1, 'relu', 16, 4, 'back_connect', 32, 'mat_add', '(', 128, 128, 3, 'sigmoid', 'mat_add', '(', 'mat_add_nodes', 'await_connection', '(', 4, 32, 4, '(', 'matmul_nodes', 5, 'mat_add', '(', 'mat_add', 16, '(', 'mat_add_nodes', 5, 16, 'tanh', 4, 128, '(', 16, '(', 64, 5, 'identity', '(', '(', 32, 'await_connection', 'mat_add_nodes', 256, 128, 'layer_norm', 'await_connection', 'flatten', 'mat_add_nodes', 'relu', '(', '(', '(', 4, 'mat_add_nodes', 'matmul', '(', 64, 32, 'relu', 256, 4, 'identity', '(', '(', 3, 16
#
# 'mat_add_nodes', 'back_connect', 'await_connection'
# ]
# network = genome.transcribe()
# print(network)
# network.fit(epochs=10, train=train_loader)
# fitness = network.evaluate(test=test_loader)
# print(f"Genome fitness: {fitness}")
#
# def sample_text(network, dataset, start_text="the ", length=200, temperature=1.0):
#     network.eval()
#     chars = list(start_text.lower())
#
#     # Convert seed to indices
#     input_seq = torch.tensor([dataset.char2idx[c] for c in chars], dtype=torch.long).unsqueeze(0).to(network.device)
#
#     for _ in range(length):
#         # Get only the last character of input if input is longer than 1 token
#         input_t = input_seq[:, -1].unsqueeze(1)
#
#         with torch.no_grad():
#             output = network(input_t)  # output shape: [batch, seq_len, vocab]
#             logits = output[:, -1, :] / temperature  # Take last time step
#
#         probs = F.softmax(logits, dim=-1).squeeze()
#         next_idx = torch.multinomial(probs, num_samples=1).item()
#         next_char = dataset.idx2char[next_idx]
#
#         chars.append(next_char)
#         input_seq = torch.cat([input_seq, torch.tensor([[next_idx]], dtype=torch.long).to(network.device)], dim=1)
#
#     return "".join(chars)
#
# # Sample text from the trained model
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.01)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.05)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.1)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.15)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.2)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.25)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.3)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.35)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.4)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.45)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.5)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.55)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.6)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.65)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.7)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.75)
# print("\nGenerated Text:\n")
# print(generated_text)
#
# generated_text = sample_text(network, dataset, start_text="the ", length=1000, temperature=0.8)
# print("\nGenerated Text:\n")
# print(generated_text)

'''Population Tests'''
# pop = Population.load("pop.pkl")
pop = gp.Population(
    size=50, # Population size (number of individuals)
    num_initial_genes=(5, 100), # Number of genes to start with for each individual
    input_shape=(1,), # Input shape
    output_shape=(len(dataset.chars),), # Output shape
    activation=None, # Activation function to use (of None, no default activation function is used)
    auto_bias=False, # Whether to automatically add bias to the network
    separate_ints=True, # Whether to separate small integers from large integers in the stacks
    mute_instructions=['batch_norm', 'conv2d', 'maxpool2d', 'avgpool2d'], # Instructions to mute
    embedding=torch.nn.Embedding(num_embeddings=len(dataset.chars), embedding_dim=50),
    embed_dim=50,
    recurrent=True,
    out_file='shakespeare_loss_50_50_2_epochs'
)
# pop.save("pop.pkl")
pop.run(
    train=train_loader,
    test=test_loader,
    generations=30, # Number of generations to run this population for
    epochs=1, # Number of epochs to train each network for
    loss_fn=torch.nn.CrossEntropyLoss(), # Loss function
    optimizer=torch.optim.Adam,
    method='epsilon-lexicase', # Selection method
    pool_size=15, # Number of individuals to select from the population for each selection into the next generation
    param_limit=50000000, # Maximum number of parameters allowed in a network
    flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
    increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
)

for genome in pop.population:
    print(genome.fitness)
    print(genome.genome)
    print("")