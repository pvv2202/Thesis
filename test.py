from torchvision import datasets, transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Define the transformation to apply to each image (convert to tensor and normalize)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Load the MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    pop = gp.Population(size=10)
    pop.initialize_population()
    pop.run(train_loader, test_loader)