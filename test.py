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
    # genome = gp.Genome(train=train_loader, test=test_loader, activation=torch.softmax)
    # genome.genome = [74, 183, 'sigmoid', 'maxpool2d', 'mat_add_nodes', 210, 'sigmoid', 'normalize', 'matmul_nodes', 86, 'normalize', 97, 'sigmoid', 'flatten', 'relu', 'matmul', 220, 'matm
    # ul_nodes', 'sigmoid', 250, 200, 'flatten', 'matmul_nodes', 'normalize', 256, 'normalize', 232, 'mat_add', 'maxpool2d', 19, 'dup', 235, 190, 183, 'mat_add_nodes', 'flatten', 'conv2d', 'dup', 70, 'relu', 'flatten', 'relu', 'sigmoid', 150, 'relu', 'mat_add_nodes', 'maxpool2d', 'relu', 'dup', 'normalize']
    # genome.genome = [99, 121, 'sigmoid', 'dup', 'mat_add', 97, 166, 16, 108, 'conv2d', 36, 'normalize', 'dup', 'maxpool2d', 'relu', 218, 129, 215, 139, 10, 'matmul_stack', 212, 'flatten', 'conv2d']
    # network = genome.transcribe()
    # print(network)
    # network.fit(epochs=1)
    # fitness = network.evaluate()
    # print(f"Genome fitness: {fitness}")

    '''Population Example'''
    # pop = Population.load("pop.pkl")
    pop = gp.Population(size=1, num_initial_genes=50, train=train_loader, test=test_loader, activation=torch.softmax)
    # pop.save("pop.pkl")
    pop.run(generations=1, epochs=1, method='tournament', pool_size=1)

    for genome in pop.population:
        print(genome.fitness)
        print(genome.genome)
        print("")