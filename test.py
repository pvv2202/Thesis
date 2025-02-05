from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gp
import torch
from interpreter import Interpreter
from instructions import Instructions
from gp import Population

if __name__ == "__main__":
    '''MNIST'''
    # # Define the transformation to apply to each image (convert to tensor and normalize)
    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.5,), (0.5,))])
    #
    # # Find device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # Load the MNIST training and test datasets
    # train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True) # drop_last=True to ensure all batches are the same size
    # test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=True) # drop_last=True to ensure all batches are the same size

    '''FASHION MNIST'''
    # # Define the transformation to apply to each image (convert to tensor and normalize)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
    # ])
    #
    # # Find device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # # Load the Fashion MNIST training and test datasets
    # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    #
    # # Create data loaders for batching
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  # drop_last=True to ensure all batches are the same size
    # test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)  # drop_last=True to ensure all batches are the same size

    '''CIFAR10'''
    # Define the transformation to apply to each image (convert to tensor and normalize)
    # CIFAR-10 images are RGB (3 channels), so normalization should be applied across all 3 channels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize R, G, B channels
    ])

    # Find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the CIFAR-10 training and test datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)  # drop_last=True to ensure all batches are the same size
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)  # drop_last=True to ensure all batches are the same size

    '''Example Network That Performs Well on MNIST'''
    # genome = gp.Genome(train=train_loader, test=test_loader, activation=torch.softmax)
    # genome.genome = [
    #     'matmul', 128, 'relu',
    #     'flatten',
    #     'maxpool2d',
    #     'conv2d', 32, 4, 'relu',  # Conv layer with 32 filters, kernel size 3
    # ]
    # network = genome.transcribe()
    # print(network)
    # # Train the network
    # genome.fitness = network.fit(epochs=20)
    # print(f"Genome fitness: {genome.fitness}")


    '''Example Network That Performs Well on CIFAR-10'''
    # interpreter = Interpreter(train=train_loader, test=test_loader, activation="relu", auto_bias=True)
    # instructions = Instructions(activation="relu")
    # genome = gp.Genome(train=train_loader, test=test_loader, interpreter=interpreter, instructions=instructions)
    # genome.genome = ['matmul', 'flatten', 'maxpool2d', 'normalize', 'relu', 'conv2d', 'normalize', 'relu,' 'conv2d', 'maxpool2d', 'normalize', 'relu', 'conv2d', 'normalize', 'relu', 'conv2d', 'maxpool2d', 'normalize', 'relu', 'conv2d', 'normalize', 'relu', 'conv2d',
    #                  128, 128, 3, 128, 3, 64, 3, 64, 3, 32, 3, 32, 3]
    # # genome.genome = ['conv2d', 'maxpool2d', 'normalize', 'relu', 'conv2d', 'normalize', 'relu', 'conv2d', 'maxpool2d', 'normalize', 'relu', 'conv2d', 'normalize', 'relu', 'conv2d',
    # #                 128, 128, 3, 128, 3, 64, 3, 64, 3, 32, 3, 32, 3]
    # network = genome.transcribe()
    # print(network)
    # # Train the network
    # network.fit(epochs=10)
    # genome.fitness = network.evaluate()
    # print(f"Fitness: {genome.fitness}")

    '''Individual Tests'''
    interpreter = Interpreter(train=train_loader, test=test_loader, activation="relu", auto_bias=True)
    instructions = Instructions(activation="relu")
    genome = gp.Genome(train=train_loader, test=test_loader, interpreter=interpreter, instructions=instructions)
    genome.genome = [
        32, 128, 5, 3, 'dup', 1, 32, 'maxpool2d', 'matmul_nodes', 'dup', 'maxpool2d', 2, 5, 4, 128, 'dup',
         'matmul_nodes', 'dup', 128, 1, 'dup', 4, 'matmul', 128, 8, 'mat_add_nodes', 2, 128, 2, 4, 'mat_add',
         'maxpool2d', 'mat_add', 'mat_add', 'flatten', 'maxpool2d', 8, 'matmul', 'conv2d', 128, 'mat_add_nodes', 'dup',
         1, 'mat_add', 'matmul', 128, 'flatten', 5, 'matmul', 'matmul_nodes', 4, 'conv2d', 'dup', 'conv2d', 128,
         'flatten', 1, 'conv2d', 'flatten', 'conv2d', 'matmul', 'matmul_nodes', 'matmul_nodes', 'maxpool2d', 'matmul',
         'matmul', 3, 'matmul_nodes', 'conv2d', 'mat_add_nodes', 'conv2d', 1, 2, 'conv2d', 'dup', 128, 'mat_add_nodes'
        # 'conv2d','conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64, 64, 64, 64, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    ]
    network = genome.transcribe()
    # network.visualize()
    print(network)
    network.fit(epochs=1)
    fitness = network.evaluate()
    print(f"Genome fitness: {fitness}")

    '''Population Example'''
    # # pop = Population.load("pop.pkl")
    # pop = gp.Population(
    #     size=75, # Population size (number of individuals)
    #     num_initial_genes=50, # Number of genes to start with for each individual
    #     train=train_loader, # Training data
    #     test=test_loader, # Testing data
    #     activation="relu", # Activation function to use (of None, no default activation function is used)
    #     auto_bias=True, # Whether to automatically add bias to the network
    #     separate_ints=True # Whether to separate small integers from large integers in the stacks
    # )
    # # pop.save("pop.pkl")
    # pop.run(
    #     generations=50, # Number of generations to run this population for
    #     epochs=1, # Number of epochs to train each network for
    #     method='epsilon-lexicase', # Selection method
    #     pool_size=75, # Number of individuals to select from the population for each selection into the next generation
    #     param_limit=50000000, # Maximum number of parameters allowed in a network
    #     flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
    #     drought=False, # Whether to use a drought mechanism that kills bad networks off early
    #     increase_epochs=False # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
    # )
    #
    # for genome in pop.population:
    #     print(genome.fitness)
    #     print(genome.genome)
    #     print("")