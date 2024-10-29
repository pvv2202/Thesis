from Random.random_gp import *
import ssl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Bypass SSL verification for downloading datasets
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True)

    unique_labels = torch.unique(train_dataset.targets)
    train_features, train_labels = next(iter(train_loader))

    # genome = Genome()
    # genome.genome = [torch.randn(784, 10, requires_grad=True), 'squeeze_tensor', 'tensor_matmul', 'tensor_flatten']
    # network = genome.transcribe(train_loader, train_loader)
    # network.fit()
    pop = Population(25, 20)
    pop.run(10000, train_loader, train_loader)


