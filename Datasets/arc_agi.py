import json
import os
import torch
import gp

def load_data(directory):
    """Load json data from directory"""
    data = []
    for file_name in os.listdir(directory):
        filepath = os.path.join(directory, file_name)
        if os.path.isfile(filepath):
            file = json.load(open(filepath, "r"))
            data.append(
                file)  # Data contains a list of dictionaries. Each item (dictionary) in the file has training and testing data

    return data


def separate_by_dim(data):
    """
    Separate data by dimensions since ARC-AGI has different dimensions for data. Separated data is a dictionary where
    keys are (input, output) dimensions and values are a list of problems with those dimension. Returns train and test
    """
    separated_data_train = {}
    separated_data_test = {}
    for problem in data:
        for i in range(2):
            if i == 0:
                key = 'train'
                d = separated_data_train
            else:
                key = 'test'
                d = separated_data_test

            for item in problem[key]:
                input_dim = ((len(item['input']), len(item['input'][0])))  # Get the dimensions of the input
                output_dim = ((len(item['output']), len(item['output'][0])))  # Get the dimensions of the output
                combined = (input_dim, output_dim)


                if combined not in d:
                    d[combined] = [item]
                else:
                    d[combined].append(item)

    return separated_data_train, separated_data_test

class ARCAGIDataset(torch.utils.data.Dataset):
    def __init__(self, data_separated):
        # Separate data by dimensions
        self.data_separated = data_separated

        # Choose dimension that has the most number of problems
        max_train_dim = (None, float('-inf'))
        for dim, problems in self.data_separated.items():
            if len(problems) > max_train_dim[1]:
                max_train_dim = (dim, len(problems))
        self.dim = max_train_dim[0]

        # Get data to use
        self.dataset = self.data_separated[self.dim]

    def __len__(self):
        """Return the number of training examples"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return a training example. Convert sample to two tensors with dtype=float32"""
        sample = self.dataset[idx]
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)  # Convert input to tensor
        output_tensor = torch.tensor(sample['output'], dtype=torch.float32)  # Convert output to tensor
        return input_tensor, output_tensor

train_directory = "data/ARC-AGI/training"
eval_directory = "data/ARC-AGI/evaluation"
train_data = load_data(train_directory)
train_data_separated, val_data_separated = separate_by_dim(train_data)
train_dataset = ARCAGIDataset(train_data_separated)
val_dataset = ARCAGIDataset(val_data_separated)

# TODO: Figure out how to handle eval

print(f"Max train dimension: {train_dataset.dim}")
input_shape = (10, 10)
output_shape = (10, 10)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

x, y = next(iter(train_loader))

'''Individual Tests'''
# interpreter = gp.Interpreter(input_shape=input_shape, output_shape=output_shape, activation=None, auto_bias=False)
# instructions = gp.Instructions(activation=None)
# genome = gp.Genome(interpreter=interpreter, instructions=instructions)
# genome.genome = ['await_connection', 'back_connect', 2, 'matmul_nodes', 'for_n', 'for_n', 'avgpool2d', 5, 'await_connection', 'mat_add', 5, 40, 4, 'avgpool2d', 'relu', 20, 'sigmoid', 'avgpool2d', '(']
# network = genome.transcribe()
# print(network)
# network.visualize()
# network.fit(epochs=1, train=train_loader)

'''Population'''
# pop = Population.load("pop.pkl")
pop = gp.Population(
    size=10000, # Population size (number of individuals)
    num_initial_genes=10, # Number of genes to start with for each individual
    input_shape=input_shape, # Training data
    output_shape=output_shape, # Testing data
    activation=None, # Activation function to use (of None, no default activation function is used)
    auto_bias=False, # Whether to automatically add bias to the network
    separate_ints=True, # Whether to separate small integers from large integers in the stacks
    mute_instructions=['await_connection', 'back_connect', 'flatten', 'transpose'], # Instructions to mute
    embedding=False,
    embed_dim=None,
    vocab_size=None,
)
# pop.save("pop.pkl")
pop.run(
    train=train_loader, # Training data
    test=val_loader, # Validation data
    generations=30, # Number of generations to run this population for
    epochs=1, # Number of epochs to train each network for
    loss_fn=torch.nn.functional.cross_entropy, # Loss function
    optimizer=torch.optim.Adam,
    method='epsilon_lexicase', # Selection method
    pool_size=15, # Number of individuals to select from the population for each selection into the next generation
    param_limit=50000000, # Maximum number of parameters allowed in a network
    flops_limit=100000000, # Maximum number of FLOPs allowed in a network
    increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
    downsample=1 # Choose whether to downsample and by how much
)

for genome in pop.population:
    print(genome.fitness)
    print(genome.genome)
    print("")

