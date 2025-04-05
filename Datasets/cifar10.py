from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import gp
import torch

if __name__ == "__main__":
    '''CIFAR-10'''
    # Define the transformation to apply to each image (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Fashion CIFAR-10 training and test datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Define dataset sizes (e.g., 80% training, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    # Split dataset into training and validation sets
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    '''Individual Tests'''
    instructions = gp.Instructions(activation=None)
    interpreter = gp.Interpreter(input_shape=(3,32,32), output_shape=(10,), instructions = instructions, activation=None, auto_bias=False)
    genome = gp.Genome(interpreter=interpreter, instructions=instructions)
    genome.genome = [
        # 1, 'mat_add', 'avgpool2d', 32, 32, '(', 'matmul_nodes', 'matmul_nodes', 'identity', 'layer_norm', 3,
        #  'matmul_nodes', 'maxpool2d', 32, 1, 32, '(', '(', 16, 'mat_add_nodes', 4, 'matmul', 'conv2d', 'layer_norm',
        #  'conv2d', 'layer_norm', 'matmul', '(', 128, 'maxpool2d', '(', 128, 128, 'batch_norm', 128, 'tanh', 256, '(',
        #  'layer_norm', 1, '(', '(', 1, '(', 128, 4, '(', '(', 'matmul_nodes', 16, '(', '(', '(', 'maxpool2d', 128, 'conv2d',
        #  128, 2, 1, 256

        # 16, 'layer_norm', 'mat_add', 256, 'mat_add', 64, 'avgpool2d', '(', 'layer_norm', 'maxpool2d', 32, 'relu', 32,
        #  64, 256, 'layer_norm', 'matmul', 'conv2d', 'conv2d', 'matmul', 'maxpool2d', '(', 'identity', 'identity', '(',
        #  128, 128, 'batch_norm', '(', 1, 'layer_norm', 256, 64, 'identity', 1, '(', 4, '(', 'tanh', '(', 'sigmoid',
        #  'matmul_nodes', 16, 64, '(', 'relu', 'maxpool2d', 'conv2d', 2, 64, 'matmul_nodes', 256

        # 16, 'layer_norm', '(', 'mat_add', 'mat_add', 64, 'avgpool2d', '(', 'matmul_nodes', 'layer_norm', 'maxpool2d',
        #  32, 32, 64, 256, 'layer_norm', 'matmul', 'conv2d', 'conv2d', 'matmul', 'maxpool2d', '(', 'identity',
        #  'identity', '(', 128, 128, 'batch_norm', '(', 1, '(', 'layer_norm', 256, 64, 1, '(', 4, '(', 'tanh', '(',
        #  'matmul_nodes', 16, 64, '(', 'relu', 'maxpool2d', 'for_n', 'conv2d', 2, 'matmul_nodes', 256
    ]
    network = genome.transcribe()
    print(network)
    network.fit(epochs=2, train=train_loader)
    fitness = network.evaluate(test=test_loader)

    '''Population Example'''
    # # pop = gp.Population.load("pop.pkl")
    # pop = gp.Population(
    #     size=75, # Population size (number of individuals)
    #     num_initial_genes=(5, 100), # Number of genes to start with for each individual
    #     input_shape=(3, 32, 32), # Training data
    #     output_shape=(10,), # Testing data
    #     activation=None, # Activation function to use (of None, no default activation function is used)
    #     auto_bias=False, # Whether to automatically add bias to the network
    #     separate_ints=True, # Whether to separate small integers from large integers in the stacks
    #     mute_instructions=['await_connection', 'back_connect'], # Instructions to mute
    #     embedding=None,
    #     embed_dim=None,
    #     out_file="redo_cifar10_lexicase_vast_50pop_20gen_1epoch"
    # )
    # # pop.save("pop.pkl")
    # pop.run(
    #     train=train_loader, # Training data
    #     test=val_loader, # Validation data
    #     generations=100, # Number of generations to run this population for
    #     epochs=1, # Number of epochs to train each network for
    #     loss_fn=torch.nn.CrossEntropyLoss(), # Loss function
    #     optimizer=torch.optim.Adam,
    #     method='epsilon_lexicase', # Selection method
    #     pool_size=15, # Number of individuals to select from the population for each selection into the next generation
    #     param_limit=50000000, # Maximum number of parameters allowed in a network
    #     flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
    #     increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
    # )
    # pop.save("pop.pkl")
    # for genome in pop.population:
    #     print(genome.fitness)
    #     print(genome.genome)
    #     print("")

# 0.5 Run:
# Best genome: [1, 'conv2d', 5, 2, 2, 'dup', 'matmul', 5, 3, 'maxpool2d', 128, 3, 256, 'maxpool2d', 'matmul_nodes', 'matmul_nodes', 32, 'matmul_nodes', 'mat_add_nodes', 'matmul_nodes', 'conv2d', 'mat_add', 'conv2d']

# 0.1 Lexicase:
# Best genome: [1, 4, 'mat_add_nodes', 1, 'matmul', 256, 4, 'avgpool2d', 3, 'mat_add_nodes', 4, 1, 'mat_add', 128, 'maxpool2d', 3, 4, 5, 5, 'conv2d', 256]

# 0.1 Tournament:
# Best genome: [64, 'avgpool2d', 64, 'avgpool2d', 4, 4, 64, 16, 'flatten', 16, 3, 'avgpool2d', 'conv2d', 'flatten', 'matmul', 'maxpool2d', 'matmul_nodes', 'conv2d', 32, 256]