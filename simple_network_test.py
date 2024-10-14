import ssl
import torch
from collections import Counter
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Bypass SSL verification for downloading datasets
    ssl._create_default_https_context = ssl._create_unverified_context

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    unique_labels = torch.unique(train_dataset.targets)
    train_features, train_labels = next(iter(train_loader))
    print(unique_labels)
    unique_train_labels = train_labels.unique()
    print(unique_train_labels)
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    for images, labels in train_loader:
        print(f"Batch of images: {images.size()}")
        print(f"Batch of labels: {labels.size()}")

    # def initialize_parameters():
    #     # Initialize weights and biases
    #     W1 = torch.randn(784, 128) * 0.01
    #     b1 = torch.zeros(128)
    #     W2 = torch.randn(128, 10) * 0.01
    #     b2 = torch.zeros(10)
    #     return W1, b1, W2, b2
    #
    #
    # def relu(x):
    #     return torch.maximum(x, torch.tensor(0.0))
    #
    #
    # def softmax(x):
    #     exp_x = torch.exp(x - x.max(dim=1, keepdim=True)[0])
    #     return exp_x / exp_x.sum(dim=1, keepdim=True)
    #
    #
    # # Training parameters
    # learning_rate = 0.01
    # epochs = 3
    #
    # # Initialize parameters
    # W1, b1, W2, b2 = initialize_parameters()
    #
    # # Training loop
    # for epoch in range(epochs):
    #     total_loss = 0
    #     for images, labels in train_loader:
    #         # Flatten images into [batch_size, 784]
    #         images = images.view(-1, 28 * 28)
    #
    #         # Forward pass
    #         z1 = images @ W1 + b1  # Linear layer
    #         a1 = relu(z1)  # Activation
    #         z2 = a1 @ W2 + b2  # Linear layer
    #         predictions = softmax(z2)  # Output
    #
    #         # Compute the loss (negative log-likelihood)
    #         loss = -torch.log(predictions[range(len(labels)), labels]).mean()
    #         total_loss += loss.item()
    #
    #         # Backpropagation
    #         grad_z2 = predictions
    #         grad_z2[range(len(labels)), labels] -= 1
    #         grad_z2 /= len(labels)
    #
    #         grad_W2 = a1.T @ grad_z2
    #         grad_b2 = grad_z2.sum(dim=0)
    #
    #         grad_a1 = grad_z2 @ W2.T
    #         grad_z1 = grad_a1.clone()
    #         grad_z1[z1 <= 0] = 0
    #
    #         grad_W1 = images.T @ grad_z1
    #         grad_b1 = grad_z1.sum(dim=0)
    #
    #         # Gradient descent step
    #         W1 -= learning_rate * grad_W1
    #         b1 -= learning_rate * grad_b1
    #         W2 -= learning_rate * grad_W2
    #         b2 -= learning_rate * grad_b2
    #
    #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
    #
    # print("Training complete!")