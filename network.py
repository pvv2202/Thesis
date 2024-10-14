import torch
import torch.nn as nn
from Instructions.instructions import Instructions
import torch.nn.functional as F
import copy
from utils import get_dim_size

class Network(nn.Module):
    '''Class For Neural Network'''
    def __init__(self, stacks, train, test):
        super(Network, self).__init__()
        self.stacks = stacks
        self.weights = stacks['params']
        self.train = train
        self.test = test

        # Get input/output shapes
        x, y = next(iter(train))
        self.input_shape = get_dim_size(x, 1) # Input shape (-1)
        self.output_shape = len(y.unique()) # Number of possible labels

        # Initialize instructions
        self.instructions = Instructions()

    def output_layer(self, mode='create'):
        '''
        Adds the weights that will project whatever the current last tensor is
        to the output dimensions. For now this just runs softmax, but could be
        adapted to support other things.
        '''
        if len(self.stack['tensor']) >= 1:
            # Pop a, get dimension
            a = self.stack['tensor'].pop()

            # If a is greater than 2D, flatten it so projection to output shape works as desired
            # TODO: There are potential issues here with respect to batching. The real potential for error is batched 1D inputs
            # TODO: For something like that we'll probably run into a bunch of problems anyway. Need to think about this
            if a.dim() > 2:
                a = torch.flatten(a, start_dim=1)

            # Get dimension of a
            a_dim = get_dim_size(a, 1) # Index -1

            if mode == 'create':
                # Create weights
                weights = torch.randn(a_dim, self.output_shape, requires_grad=True)
                self.stack['params'].append(weights)
            elif mode == 'weights':
                # Get weights
                weights = self.stack['params'].pop()

            # Multiply a by weights and return. Should now have dimension of output_shape
            return torch.matmul(a, weights)

    def forward(self, x, mode='create'):
        '''Forward Pass'''
        # Maintain a count of pushed tensors. This is for the input logic. Maybe would be smart to remove this?
        push_tensor_count = sum([1 for instruction in self.stacks['exec'] if instruction == 'push_tensor'])
        pushed_tensors = 0

        # Add input data to tensor stack
        self.stacks['tensor'].append(x)

        # Run the program. Iterate over the exec stack
        for instruction in self.stacks['exec']:
            # Special cases for pushing a tensor since these are our weights
            if instruction == 'push_tensor':
                # If this is the last tensor we're going to push, make it compatible with input
                if pushed_tensors == push_tensor_count-1:
                    self.instructions(self.stacks, instruction, mode='input', input_shape=self.input_shape)
                # Otherwise, just push the tensor as normal
                else:
                    self.instructions(self.stacks, instruction, mode=mode)
            # If not pushing a tensor, just execute the instruction
            elif instruction in self.instructions.instructions:
                self.instructions(self.stacks, instruction)

        # Run the output layer
        self.output_layer(mode)

        if mode == 'create':
            # Set weights to be params
            self.weights = copy.deepcopy(self.stacks['params'])
        elif mode == 'weights':
            # Reset params to original state (copying weights)
            self.stacks['params'] = copy.deepcopy(self.weights)

        # Return final tensor on the stack as output
        return self.stacks['tensor'][-1]

    def compute_loss(self, output_tensor, batch_labels, loss_function='cross_entropy'):
        '''Compute the loss with respect to the target'''
        if loss_function == 'cross_entropy':
            # Compute the loss (cross-entropy). PyTorch's cross-entropy function combines softmax and cross-entropy
            loss = F.cross_entropy(output_tensor, batch_labels)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        return loss

    def fit(self, epochs=3, optimizer_name='adam', loss='cross_entropy', learning_rate=0.01):
        '''Fit the network'''
        # Choose optimizer
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.stacks['params'], lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.stacks['params'], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # TODO: There's a chance that networks which fail to execute will have no loss. Keep this in mind
        for epoch in range(epochs):
            # Training loop
            total_loss = 0.0  # Accumulate loss over the epoch
            for batch_num, batch_inputs, batch_labels in enumerate(self.train):
                optimizer.zero_grad()  # Reset gradients

                # Forward pass: use batch tensors as input to the model
                if epoch == 0 and batch_num == 0:
                    # If on the very first pass, we need to create the weights for the first time
                    output_tensor = self.forward(batch_inputs, mode='create')
                else:
                    # If weights have been created, we can just use them
                    output_tensor = self.forward(batch_inputs, mode='weights')

                # TODO: I think it's possible for the output tensor to have an incorrect shape

                # Compute loss with respect to the target (batch_labels)
                loss = self.compute_loss(output_tensor, batch_labels)

                # Backpropagation
                loss.backward()

                # Update parameters
                optimizer.step()

                total_loss += loss.item() * batch_inputs.size(0)  # Multiply by batch size

            average_loss = total_loss / len(self.train.dataset)
            print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

            return average_loss