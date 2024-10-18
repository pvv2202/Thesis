import torch
import torch.nn as nn
from instructions import Instructions
import torch.nn.functional as F
import copy
from utils import get_dim_size, broadcastable

# TODO: Maybe a different type of copy will allow not resetting params?

class Network(nn.Module):
    '''Class For Neural Network'''
    def __init__(self, stacks, train, test):
        super(Network, self).__init__()
        self.stacks = stacks
        self.network = copy.deepcopy(stacks)
        self.train = train
        self.test = test

        # Get input/output shapes
        x, y = next(iter(train))
        self.batch_size = x.size(0) # Batch size
        self.input_shape = get_dim_size(x, 1) # Input shape (-1)
        self.output_shape = len(y.unique()) # Number of possible labels

        # Initialize instructions
        self.instructions = Instructions()

    # def output_layer(self, create=True):
    #     '''
    #     Adds the weights that will project whatever the current last tensor is
    #     to the output dimensions. For now this just runs softmax, but could be
    #     adapted to support other things.
    #     '''
    #
    #     if len(self.stacks['tensor']) >= 1:
    #
    #         if not create:
    #             print(self.stacks['tensor'])
    #
    #         # Pop a, get dimension
    #         a = self.stacks['tensor'].pop()
    #
    #         # Get dimension of a
    #         a_dim = get_dim_size(a, 1)  # Index -1
    #
    #         # If a is greater than 2D, flatten it so projection to output shape works as desired
    #         # TODO: There are potential issues here with respect to batching. The real potential for error is batched 1D inputs
    #         # TODO: For something like that we'll probably run into a bunch of problems anyway. Need to think about this
    #         if a_dim > 2:
    #             a = torch.flatten(a, start_dim=0, end_dim=-1)
    #             a_dim = get_dim_size(a, 1)
    #
    #         weights = None
    #         if create:
    #             # Create weights
    #             weights = torch.randn(a_dim, self.output_shape, requires_grad=True)
    #             self.stacks['params'].append(weights)
    #         elif len(self.stacks['params']) >= 1:
    #             # Get weights
    #             weights = self.stacks['params'].pop(0)
    #
    #         # Multiply a by weights and add to tensor stack. Should now have dimension of output_shape
    #         if weights is not None:
    #             if broadcastable(a, weights):
    #                 self.stacks['tensor'].append(torch.matmul(a, weights))

    def forward(self, x):
        '''Forward Pass'''
        # Reset the stacks to the original state
        self.stacks = copy.deepcopy(self.network)

        # Add input data to tensor stack
        self.stacks['tensor'].append(x)

        # Run the program. Iterate over the exec stack
        while len(self.stacks['exec']) > 0:
            instruction = self.stacks['exec'].pop()
            self.instructions(self.stacks, instruction)

        # Return final tensor on the stack as output
        if len(self.stacks['tensor']) >= 1:
            if self.stacks['tensor'][-1].size() == (self.batch_size, self.output_shape):
                return self.stacks['tensor'][-1]
        else:
            return None

    def compute_loss(self, output_tensor, batch_labels, loss_function='cross_entropy'):
        '''Compute the loss with respect to the target'''
        if loss_function == 'cross_entropy':
            # Compute the loss (cross-entropy). PyTorch's cross-entropy function combines softmax and cross-entropy
            loss = F.cross_entropy(output_tensor, batch_labels)
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")

        return loss

    def fit(self, epochs=3, optimizer_name='adam', loss_function='cross_entropy', learning_rate=0.01):
        '''Fit the network'''
        # If we have no params, we can't train so return infinity
        if len(self.stacks['params']) == 0:
            print("No parameters to train")
            return float('inf')

        # Choose optimizer. Give it params so it knows which weights to update
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.stacks['params'], lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.stacks['params'], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        for epoch in range(epochs):
            # Training loop
            total_loss = 0.0  # Accumulate loss over the epoch
            for batch_inputs, batch_labels in self.train:
                optimizer.zero_grad()  # Reset gradients

                output_tensor = self.forward(batch_inputs)

                # If the network doesn't run, set loss to infinity and return
                if output_tensor is None:
                    print("Output dimension incompatible")
                    return float('inf')

                # Compute loss with respect to the target (batch_labels)
                loss = self.compute_loss(output_tensor, batch_labels, loss_function=loss_function)

                # Backpropagation (compute gradients for each tensor in parameters w.r.t loss). PyTorch's autograd constructs and
                # Maintains the computation graph
                loss.backward()

                # Update parameters (just adds the gradients for each weight to the respective weights)
                optimizer.step()

                total_loss += loss.item() * batch_inputs.size(0)  # Multiply by batch size

            average_loss = total_loss / len(self.train.dataset) # Average loss across all batches
            print(f"Epoch {epoch + 1}, Loss: {average_loss:.4f}")

            return average_loss