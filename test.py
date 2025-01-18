from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gp
import torch
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
    # #TODO: Right now, this requires fixed batch sizes. Ideally, it should be able to handle variable batch sizes. Network object could take a "batch" field
    # # Create data loaders for batching
    # train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True) # drop_last=True to ensure all batches are the same size
    # test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=True) # drop_last=True to ensure all batches are the same size

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
    # genome = gp.Genome(train=train_loader, test=test_loader, activation=torch.softmax)
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
    # genome = gp.Genome(train=train_loader, test=test_loader, activation='relu')
    # genome.genome = [
    #     'flatten', 'mat_add_nodes', 'normalize', 256, 3, 64, 'conv2d', 1, 64, 4, 'mat_add', 'flatten', 4, 4, 1, 'normalize', 'matmul', 4, 'matmul', 3, 32, 'mat_add_nodes', 'conv2d', 'mat_add', 'matmul', 128, 'matmul', 'mat_add_nodes', 'normalize', 'mat_add', 'normalize', 'maxpool2d', 'normalize', 'normalize', 128, 16, 64, 'matmul', 16, 2, 'matmul_nodes', 8, 4, 'mat_add', 128, 'flatten', 'mat_add', 3, 'dup', 'maxpool2d', 'matmul'
    # ]
    # network = genome.transcribe()
    # print(network)
    # network.fit(epochs=10)
    # fitness = network.evaluate()
    # print(f"Genome fitness: {fitness}")

    '''Population Example'''
    pop = Population.load("pop.pkl")
    # pop = gp.Population(size=2, num_initial_genes=50, train=train_loader, test=test_loader, activation="relu", auto_bias=True)
    # pop.save("pop.pkl")
    pop.run(generations=2, epochs=1, method='tournament', pool_size=8, param_limit=500000)

    for genome in pop.population:
        print(genome.fitness)
        print(genome.genome)
        print("")