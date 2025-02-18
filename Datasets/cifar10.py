from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import gp
import torch

if __name__ == "__main__":
    '''CIFAR-10'''
    # Define the transformation to apply to each image (convert to tensor and normalize)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize grayscale images
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
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    '''Individual Tests'''
    # interpreter = gp.Interpreter(input_shape=(3,32,32), output_shape=(10,), activation="relu", auto_bias=True)
    # instructions = gp.Instructions(activation="relu")
    # genome = gp.Genome(interpreter=interpreter, instructions=instructions)
    # genome.genome = [
    #     3, 3, 'mat_add', 8, 'matmul', 2, 128, 'conv2d', 'dup', 'avgpool2d', 'maxpool2d', 16, 'maxpool2d', 'conv2d', 4, 2, 16, 256, 'mat_add', 64
    #     # 'matmul', 'matmul', 'matmul', 'matmul', 'matmul', 'matmul', 'flatten', 'maxpool2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'conv2d', 'maxpool2d', 'conv2d',
    #     # 32, 64, 128, 256, 512, 1024, 512, 512, 256, 256, 128, 128, 64, 64, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
    # ]
    # network = genome.transcribe()
    # print(network)
    # network.fit(epochs=10, train=train_loader, test=val_loader)
    # network.visualize()
    # fitness = network.evaluate(test=test_loader)

    '''Population Example'''
    # pop = Population.load("pop.pkl")
    pop = gp.Population(
        size=50, # Population size (number of individuals)
        num_initial_genes=20, # Number of genes to start with for each individual
        input_shape=(3, 32, 32), # Training data
        output_shape=(10,), # Testing data
        activation="relu", # Activation function to use (of None, no default activation function is used)
        auto_bias=True, # Whether to automatically add bias to the network
        separate_ints=True, # Whether to separate small integers from large integers in the stacks
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
        method='tournament', # Selection method
        pool_size=15, # Number of individuals to select from the population for each selection into the next generation
        param_limit=50000000, # Maximum number of parameters allowed in a network
        flops_limit=5000000000, # Maximum number of FLOPs allowed in a network
        drought=False, # Whether to use a drought mechanism that kills bad networks off early
        increase_epochs=False, # Whether to increase the number of epochs (can also be a fraction of epochs) trained based on the generation
        downsample=0.5 # Choose whether to downsample and by how much
    )

    for genome in pop.population:
        print(genome.fitness)
        print(genome.genome)
        print("")