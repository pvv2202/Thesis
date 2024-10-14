import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

'''Default Values'''
# TODO: Do I even want a bunch of default values? Or should I just have noop?
# TODO: Should I pop and then check if valid or check if valid and then pop?

# Mapping of instruction strings to method references
valid_instructions = [
    # Tensor operations
    'push_tensor',
    'duplicate_tensor',
    'transpose_tensor',
    'tensor_matmul',
    'tensor_matmul_duplicate',
    'tensor_max_pool',
    'tensor_convolve',
    'tensor_normalize',
    'tensor_add',
    'tensor_sub',
    'tensor_relu',
    'tensor_sigmoid',
    'tensor_flatten',
    'tensor_divide_float',

    # Int operations
    'int_add',
    'int_mult',
    'int_divide',
    'int_sqrt',

    # Float operations
    'float_add',
    'float_mult',
    'float_divide',
    'float_sqrt',

    # Misc operations
    'add_previous_layer'
]

class Interpreter_Old(nn.Module):
    '''Push Interpreter'''
    def __init__(self, train, test, batch_size=64):
        super(Interpreter_Old, self).__init__()
        # Initialize stacks
        self.stacks = {
            'int': [], # Really just Natural numbers
            'float': [],
            'bool': [],
            'str': [],
            'tensor': [],
            'exec': []
        }

        self.network = {} # Network is stored as a dictionary of stacks

        # Initialize data, batch size
        self.train = train
        self.test = test
        self.batch_size = batch_size

        # Store parameters for optimization
        self.parameters = []

        # Dimensions of the output of the previous layer. Initially it's the number of features in the input tensor
        self.previous_layer = next(iter(train))[0].size()[-1]

        # Mapping of instruction strings to method references
        self.instructions_map = {
            # Tensor operations
            'push_tensor': self.push_tensor,
            'duplicate_tensor': self.duplicate_tensor,
            'transpose_tensor': self.transpose_tensor,
            'tensor_matmul': self.tensor_matmul,
            'tensor_matmul_duplicate': self.tensor_matmul_duplicate,
            'tensor_max_pool': self.tensor_max_pool,
            'tensor_convolve': self.tensor_convolve,
            'tensor_normalize': self.tensor_normalize,
            'tensor_add': self.tensor_add,
            'tensor_sub': self.tensor_sub,
            'tensor_relu': self.tensor_relu,
            'tensor_sigmoid': self.tensor_sigmoid,
            'tensor_flatten': self.tensor_flatten,
            'tensor_divide_float': self.tensor_divide,

            # NN Operations
            'output_layer': self.output_layer,
            'compute_loss': self.compute_loss,
            'fit': self.fit,

            # Int operations
            'int_add': self.int_add,
            'int_mult': self.int_mult,
            'int_divide': self.int_divide,
            'int_sqrt': self.int_sqrt,

            # Float operations
            'float_add': self.float_add,
            'float_mult': self.float_mult,
            'float_divide': self.float_divide,
            'float_sqrt': self.float_sqrt,

            # Misc operations
            'add_previous_layer': self.add_previous_layer
        }

        """Dense layer would be like 1x5 x 5x4 to get 1x4. Attention is obviously all matrix mult."""

    #TODO: Maybe add a function to project 2D/3D to 4D tensor for convolutions/pooling?

    '''
    Misc Operations
    '''
    def add_previous_layer(self):
        '''Adds the previous layer's dimension to the int stack'''
        self.stacks['int'].append(self.previous_layer)

    '''
    Int Operations
    '''
    def int_add(self):
        '''Adds two integers'''
        if len(self.stacks['int']) >= 2:
            a = self.stacks['int'].pop()
            b = self.stacks['int'].pop()
            self.stacks['int'].append(a + b)

    def int_mult(self):
        '''Multiplies two integers'''
        if len(self.stacks['int']) >= 2:
            a = self.stacks['int'].pop()
            b = self.stacks['int'].pop()
            self.stacks['int'].append(a * b)

    def int_divide(self):
        '''Divides two integers'''
        if len(self.stacks['int']) >= 2:
            a = self.stacks['int'].pop()
            b = self.stacks['int'].pop()
            self.stacks['int'].append(a // b)

    def int_sqrt(self):
        '''Takes the square root of an integer'''
        if len(self.stacks['int']) >= 1:
            a = self.stacks['int'].pop()
            self.stacks['int'].append(a ** 0.5)

    '''
    Float Operations
    '''
    def float_add(self):
        '''Adds two floats'''
        if len(self.stacks['float']) >= 2:
            a = self.stacks['float'].pop()
            b = self.stacks['float'].pop()
            self.stacks['float'].append(a + b)

    def float_mult(self):
        '''Multiplies two floats'''
        if len(self.stacks['float']) >= 2:
            a = self.stacks['float'].pop()
            b = self.stacks['float'].pop()
            self.stacks['float'].append(a * b)

    def float_divide(self):
        '''Divides two floats'''
        if len(self.stacks['float']) >= 2:
            a = self.stacks['float'].pop()
            b = self.stacks['float'].pop()
            self.stacks['float'].append(a / b)

    def float_sqrt(self):
        '''Takes the square root of a float'''
        if len(self.stacks['float']) >= 1:
            a = self.stacks['float'].pop()
            self.stacks['float'].append(a ** 0.5)

    '''
    Tensor Stack Operations
    '''
    def push_tensor(self):
        '''Adds a tensor to the stack'''
        # If int_stack is empty, do nothing
        if len(self.stacks['int']) > 0:
            self.stacks['tensor'].append(torch.randn(self.previous_layer, self.stacks['int'].pop(), requires_grad=True)) # Use previous layer to be compatible
            self.parameters.append(self.stacks['tensor'][-1]) # Add to parameters for optimization
            self.previous_layer = self.stacks['tensor'][-1].size()[-1] # Update previous layer to be the number of features in the tensor

    def duplicate_tensor(self):
        '''Duplicates the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'][-1]
            self.stacks['tensor'].append(a)

    def transpose_tensor(self):
        '''Transposes the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'].pop()
            result = torch.transpose(a, 0, 1)
            self.stacks['tensor'].append(result)
            self.previous_layer = result.size()[-1]

    '''
    Tensor Operations
    '''
    # TODO: Think about ways to deal with misaligned matrices and how we can smooth any weirdness down the line
    # TODO: For GP, we probably want to have our functions be kind of "smart" i.e. if we specify adding a matmul or something, ensuring that we add a tensor that can handle that? Or just whenever we add a tensor we add one that is compatible with the previous one?
    def tensor_matmul(self):
        '''
        Performs a matrix multiplication on the top two tensors on the stack. Pytorch's matmul function
        automatically supports broadcasting (i.e. multiplying tensors of different sizes for batch/channel dimensions)
        '''
        if len(self.stacks['tensor']) >= 2:
            b = self.stacks['tensor'].pop()
            a = self.stacks['tensor'].pop()

            # Check if sizes are compatible. If not, just return
            if a.size()[-1] != b.size()[-2]:
                return

            result = torch.matmul(a, b)
            self.stacks['tensor'].append(result)
            self.previous_layer = result.size()[-1]

    def tensor_matmul_duplicate(self):
        '''Performs a matrix multiplication on the top tensor on the stack but duplicates a'''
        if len(self.stacks['tensor']) >= 1:
            b = self.stacks['tensor'].pop()
            a = self.stacks['tensor'].pop()

            # Check if sizes are compatible. If not, just return
            if a.size()[-1] != b.size()[-2]:
                return

            result = torch.matmul(a, b)
            self.stacks['tensor'].append(result)
            self.stacks['tensor'].append(a)
            self.previous_layer = result.size()[-1]

    def tensor_max_pool(self):
        '''Performs a pooling operation on the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            # If 1-dimensional, do nothing
            if self.stacks['tensor'][-1].dim() == 1:
                return

            # Otherwise, pool
            a = self.stacks['tensor'].pop()

            # Adjust a's shape to be compatible with torch.nn.functional.max_pool2d which expects 4D tensors (batch, channel, height, width)
            if a.dim() == 2:
                a = a.unsqueeze(0).unsqueeze(0)
            elif a.dim() == 3:
                a = a.unsqueeze(0)

            # Pop kernel size and stride from stack
            if len(self.stacks['int']) > 1:
                kernel_size = self.stacks['int'].pop()
                stride = self.stacks['int'].pop() # Don't need constraints on stride. If huge, we will just have 1x1 which should be weeded out via evolution anyway
                if kernel_size < min(a.size(2), a.size(3)): # Only continue if kernel size is compatible (i.e. < height and width of input)
                    result = nn.functional.max_pool2d(a, kernel_size=kernel_size, stride=stride)
                    self.stacks['tensor'].append(result)
                    self.previous_layer = result.size()[-1]

    def tensor_convolve(self):
        '''
        Performs a convolution on the top two tensors on the stack
        Here this is defined as multiplying along rows and columns as much as it can.
        '''
        if len(self.stacks['tensor']) >= 2:
            # TODO: Think harder about this. Would there ever be an advantage to projecting a 1D tensor to a 2D tensor and convolving? Maybe include the option and see?
            # If either is 1-dimensional, do nothing
            if self.stacks['tensor'][-1].dim() == 1 or self.stacks['tensor'][-2].dim() == 1:
                return

            b = self.stacks['tensor'].pop() # b will be the kernel
            a = self.stacks['tensor'].pop()

            # Adjust a's shape to be compatible with nn.functional.conv2d which expects 4D tensors (batch, channel, height, width)
            if a.dim() == 2:
                a = a.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, aH, aW]
            elif a.dim() == 3:
                a = a.unsqueeze(0)  # Shape: [1, channels, aH, aW]

            # Adjust b's shape to be compatible with nn.functional.conv2d which expects 4D tensors (out_channels, in_channels, kernel_height, kernel_width)
            # Here, out_channels determines how many channels will be in the output tensor. The rest are dimensions of the kernel
            if b.dim() == 2:
                b = b.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, bH, bW]
                b = b.expand(a.size(1), a.size(1), -1, -1)  # Shape: [in_channels, in_channels, bH, bW]
            elif b.dim() == 3:
                b = b.unsqueeze(1)  # Shape: [C_out, 1, bH, bW]
                b = b.expand(-1, a.size(1), -1, -1)  # Shape: [out_channels, in_channels, bH, bW]

            # TODO: Have 2 separate functions, one that pads and one that doesn't?
            # Calculate padding to achieve 'same' convolution (output size matches input size)
            bH, bW = b.size(2), b.size(3)
            padding = (bH // 2, bW // 2)

            # Perform convolution
            result = nn.functional.conv2d(a, b, padding=padding)

            self.stacks['tensor'].append(result)
            self.previous_layer = result.size()[-1]

    def tensor_normalize(self):
        '''Normalizes the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'].pop()
            result = nn.functional.normalize(a)
            self.stacks['tensor'].append(result)

    def tensor_add(self):
        '''
        Performs an addition on the top two tensors on the stack. Pytorch automatically supports broadcasting
        so we don't need to worry about tensor sizes being the same
        '''
        if len(self.stacks['tensor']) >= 2:
            a = self.stacks['tensor'].pop()
            b = self.stacks['tensor'].pop()
            result = torch.add(a, b)
            self.stacks['tensor'].append(result)

    def tensor_sub(self):
        '''
        Performs a subtraction on the top two tensors on the stack. Pytorch automatically supports broadcasting
        so we don't need to worry about tensor sizes being the same
        '''
        if len(self.stacks['tensor']) >= 2:
            a = self.stacks['tensor'].pop()
            b = self.stacks['tensor'].pop()
            result = torch.sub(a, b)
            self.stacks['tensor'].append(result)

    def tensor_relu(self):
        '''Performs a ReLU activation on the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'].pop()
            result = torch.relu(a)
            self.stacks['tensor'].append(result)

    def tensor_sigmoid(self):
        '''Performs a Sigmoid activation on the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'].pop()
            result = torch.sigmoid(a)
            self.stacks['tensor'].append(result)

    def tensor_flatten(self):
        '''Flattens the top tensor on the stack'''
        if len(self.stacks['tensor']) >= 1:
            a = self.stacks['tensor'].pop()
            result = torch.flatten(a)
            self.stacks['tensor'].append(result)
            self.previous_layer = result.size()[-1]

    def tensor_divide(self):
        '''Divides all elements of the top tensor on the stack by a float from the float stack'''
        if len(self.stacks['tensor']) >= 1 and len(self.stacks['float']) >= 1:
            tensor = self.stacks['tensor'].pop()
            divisor = self.stacks['float'].pop()

            result = tensor / divisor
            self.stacks['tensor'].append(result)

    '''
    NN Operations
    '''
    def output_layer(self):
        '''
        Adds an output layer by transforming the top tensor to have dimensions `dim` and applies
        an activation function.
        '''
        if len(self.stacks['tensor']) >= 1:
            # Get dimension of labels
            first_batch = next(iter(self.train))
            labels_sample = first_batch[1]
            dim = labels_sample.size()[-1]  # Extract last dimension of the labels (e.g., num_classes)

            a = self.stacks['tensor'].pop()

            if a.dim() == 1:
                a = a.unsqueeze(0)  # Shape: [1, features]
            if a.dim() > 2:
                a = a.view(a.size(0), -1)  # Shape: [batch_size, num_features]

            num_features = a.size()[-1]

            # Correctly define weight and bias with integer dimensions
            weight = torch.randn(num_features, dim, requires_grad=True)
            bias = torch.randn(dim, requires_grad=True)

            self.parameters.extend([weight, bias])

            output = torch.matmul(a, weight) + bias  # Shape: [batch_size, dim]

            # Apply activation function, probably softmax
            output = torch.softmax(output, dim=1)

            # Push the resulting tensor back onto the stack as output
            self.stacks['tensor'].append(output)

    def compute_loss(self, tensor, target, loss='mse'):
        # TODO: Allow for a variety of loss functions
        # Example loss: mean squared error with target tensor
        loss = nn.functional.mse_loss(tensor, target)
        return loss

    def fit(self, num_epochs=5, learning_rate=0.01, optimizer_type='adam'):
        # Choose optimizer
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        average_loss = float('inf')

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0.0  # Accumulate loss over the epoch
            for batch_inputs, batch_labels in self.train:
                optimizer.zero_grad()  # Reset gradients

                # Forward pass: use batch tensors as input to the model
                output_tensor = self.forward(batch_inputs)

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

    def forward(self, input_tensor):
        """
        Forward pass through the model.
        """
        # Copy the network to avoid modifying the original
        self.stacks = copy.deepcopy(self.network)
        # Append input tensor to tensor stack
        self.stacks['tensor'].append(input_tensor)
        # Run the interpreter
        self.run()

    """
    Interpreter Methods
    """
    def read_genome(self, genome):
        '''Reads a genome and processes it into the stacks'''
        for instruction in genome:
            if type(instruction) == float:
                self.stacks['float'].append(instruction)
            elif type(instruction) == int:
                self.stacks['int'].append(instruction)
            elif type(instruction) == bool:
                self.stacks['bool'].append(instruction)
            elif type(instruction) == str:
                if instruction in self.instructions_map:
                    self.stacks['exec'].append(self.instructions_map[instruction])
                else:
                    self.stacks['str'].append(instruction)
            elif type(instruction) == torch.Tensor:
                self.stacks['tensor'].append(instruction)

        # Assign network to be a copy of the stacks for future forward passes
        self.network = copy.deepcopy(self.stacks)

    def run(self):
        '''Runs the interpreter'''
        while self.stacks['exec']:
            instruction = self.stacks['exec'].pop(0)
            if callable(instruction):
                instruction()
            else:
                raise ValueError(f'Invalid instruction: {instruction}')