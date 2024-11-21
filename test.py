from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import gp
import torch

if __name__ == "__main__":
    # Define the transformation to apply to each image (convert to tensor and normalize)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True) # drop_last=True to ensure all batches are the same size
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, drop_last=True) # drop_last=True to ensure all batches are the same size

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

    '''Testing for matmul_stack'''
    # genome = gp.Genome(train=train_loader, test=test_loader, activation=torch.softmax)
    # genome.genome = ['flatten', 'flatten', 'relu', 213, 238, 185, 'matmul', 245, 32, 209, 'flatten', 'sigmoid', 'mat_add_stack', 'flatten', 47, 166, 'matmul_dup', 'mat_add_dup', 'matmul_dup', 153, 115, 'mat_add_stack', 'matmul_stack', 248]
    # network = genome.transcribe()
    # print(network)
    # network.fit(epochs=1)
    # fitness = network.evaluate()
    # print(f"Genome fitness: {fitness}")

    '''Population Example'''
   