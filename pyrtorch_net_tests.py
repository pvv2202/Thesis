import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchinfo import summary
import torch.nn.init as init

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
train_loader = DataLoader(dataset=train_dataset, batch_size=64,
                          shuffle=True)  # drop_last=True to ensure all batches are the same size
test_loader = DataLoader(dataset=test_dataset, batch_size=64,
                         shuffle=False)  # drop_last=True to ensure all batches are the same size

# Create a simple network using default pytorch functions
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*32*32, 10),
)

summary(model, input_size=(64, 3, 32, 32))

weight = torch.empty((3*32*32, 10), requires_grad=True)
bias = torch.empty((10), requires_grad=True)
init.xavier_uniform_(weight)
init.zeros_(bias)

params = [weight, bias]

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer1 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer2 = torch.optim.SGD(params, lr=0.001, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        # forward + backward + optimize for model 1
        outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss1.backward()
        optimizer1.step()

        # forward + backward + optimize for model 2
        flat = torch.flatten(inputs, start_dim=1)
        output = torch.matmul(flat, weight)
        #output = torch.add(mul, bias)
        loss2 = criterion(output, labels)
        loss2.backward()
        optimizer2.step()

# Get accuracy on test set
correct1 = 0
total = 0
correct2 = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        total += labels.size(0)

        # Get accuracy for model 1
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct1 += (predicted == labels).sum().item()

        # Get accuracy for model 2
        flat = torch.flatten(images, start_dim=1)
        output = torch.matmul(flat, weight)
        # output = torch.add(mul, bias)
        _, predicted = torch.max(output.data, 1)
        correct2 += (predicted == labels).sum().item()

print(f'Accuracy 1: {100 * correct1 / total}%')
print(f'Accuracy 2: {100 * correct2 / total}%')

