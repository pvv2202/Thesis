from alice import TextDataset
from torch.utils.data import DataLoader, random_split
import gp
import torch

# Load dataset
dataset = TextDataset(file_path="data/tiny_shakespeare.txt", seq_length=20)

# Split into train (80%) and test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders. Need drop_last true to ensure hidden state at t-1 will be compatible with input
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

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
    mute_instructions=['batch_norm'], # Instructions to mute
    embedding=torch.nn.Embedding(num_embeddings=len(dataset.chars), embedding_dim=50),
    embed_dim=50,
    recurrent=True,
    out_file='shakespeare_loss_75_100'
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