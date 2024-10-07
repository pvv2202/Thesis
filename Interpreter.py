import torch

class Interpreter:
    '''Push Interpreter'''
    def __init__(self, X_train, y_train, X_test, y_test):
        # Initialize stacks
        self.int_stack = []
        self.float_stack = []
        self.bool_stack = []
        self.str_stack = []
        self.tensor_stack = []
        self.exec_stack = []

        # Initialize data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Store parameters for optimization
        self.parameters = []

        # Mapping of instruction strings to method references
        self.instructions_map = {
            'add': self.tensor_add,
            'sub': self.tensor_sub,
            'relu': self.tensor_relu,
            'matmul': self.tensor_matmul,
            'sigmoid': self.tensor_sigmoid,
            'normalize': self.tensor_normalize,
            'convolve': self.tensor_convolve,
            'flatten': self.tensor_flatten,
            'fit': self.fit
        }

        """Dense layer would be like 1x5 x 5x4 to get 1x4. Attention is obviously all matrix mult.
        Conv layer is more complicated. Maybe need to define "complete mult" function or something that convolves. 
        Also need to figure out how to implement reshaping and padding and stuff. 
        """

    '''
    Tensor Operations
    '''

    #TODO: For GP, we probably want to have our functions be kind of "smart" i.e. if we specify adding a matmul or something, ensuring that we add a tensor that can handle that? Or just whenever we add a tensor we add one that is compatible with the previous one?
    def tensor_matmul(self):
        '''
        Performs a matrix multiplication on the top two tensors on the stack. Pytorch's matmul function
        automatically supports broadcasting (i.e. multiplying tensors of different sizes for batch/channel dimensions)
        '''
        if len(self.tensor_stack) >= 2:
            a = self.tensor_stack.pop()
            b = self.tensor_stack.pop()

            # Check if sizes are compatible. If not, just return
            if a.size()[-1] != b.size()[-2]:
                return

            result = torch.matmul(a, b)
            self.tensor_stack.append(result)

    def tensor_convolve(self):
        '''
        Performs a convolution on the top two tensors on the stack
        Here this is defined as multiplying along rows and columns as much as it can.
        '''
        if len(self.tensor_stack) >= 2:
            a = self.tensor_stack.pop()
            b = self.tensor_stack.pop() # b will be the kernel

            # If either is 1-dimensional, do nothing
            if a.dim() == 1 or b.dim() == 1:
                return #TODO: Think harder about this. Would there ever be an advantage to projecting a 1D tensor to a 2D tensor and convolving? Maybe include the option and see?

            if a.dim() == 2:
                a = a.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, aH, aW]
            elif a.dim() == 3:
                a = a.unsqueeze(0)  # Shape: [1, C, aH, aW]

            if b.dim() == 2:
                b = b.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, bH, bW]
                b = b.expand(a.size(1), a.size(1), -1, -1)  # Shape: [C_in, C_in, bH, bW]
            elif b.dim() == 3:
                b = b.unsqueeze(1)  # Shape: [C_out, 1, bH, bW]
                b = b.expand(-1, a.size(1), -1, -1)  # Shape: [C_out, C_in, bH, bW]

            # Calculate padding to achieve 'same' convolution (output size matches input size)
            bH, bW = b.size(2), b.size(3)
            padding = (bH // 2, bW // 2)

            # Perform convolution
            result = torch.nn.functional.conv2d(a, b, padding=padding)

            self.tensor_stack.append(result)

    def tensor_normalize(self):
        '''Normalizes the top tensor on the stack'''
        if len(self.tensor_stack) >= 1:
            a = self.tensor_stack.pop()
            result = torch.nn.functional.normalize(a)
            self.tensor_stack.append(result)

    def tensor_add(self):
        '''Performs an addition on the top two tensors on the stack'''
        if len(self.tensor_stack) >= 2:
            a = self.tensor_stack.pop()
            b = self.tensor_stack.pop()
            result = torch.add(a, b)
            self.tensor_stack.append(result)

    def tensor_sub(self):
        '''Performs a subtraction on the top two tensors on the stack'''
        if len(self.tensor_stack) >= 2:
            a = self.tensor_stack.pop()
            b = self.tensor_stack.pop()
            result = torch.sub(a, b)
            self.tensor_stack.append(result)

    def tensor_relu(self):
        '''Performs a ReLU activation on the top tensor on the stack'''
        if len(self.tensor_stack) >= 1:
            a = self.tensor_stack.pop()
            result = torch.relu(a)
            self.tensor_stack.append(result)

    def tensor_sigmoid(self):
        '''Performs a Sigmoid activation on the top tensor on the stack'''
        if len(self.tensor_stack) >= 1:
            a = self.tensor_stack.pop()
            result = torch.sigmoid(a)
            self.tensor_stack.append(result)

    def tensor_flatten(self):
        '''Flattens the top tensor on the stack'''
        if len(self.tensor_stack) >= 1:
            a = self.tensor_stack.pop()
            result = torch.flatten(a)
            self.tensor_stack.append(result)

    '''NN Operations'''
    def compute_loss(self, tensor):
        # Example loss: mean squared error with target tensor
        target = torch.zeros_like(tensor) # This is just a placeholder
        loss = torch.nn.functional.mse_loss(tensor, target)
        return loss

    def fit(self, num_epochs=5):
        optimizer = torch.optim.Adam(self.parameters, lr=0.01)

        for epoch in range(num_epochs):
            optimizer.zero_grad()  # Reset gradients

            # Assume the result is on top of the tensor stack
            output_tensor = self.tensor_stack.pop()

            # Compute loss
            loss = self.compute_loss(output_tensor)

            # Backpropagation
            loss.backward()

            # Update parameters
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    """
    Interpreter Methods
    """

    def read_genome(self, genome):
        '''Reads a genome and processes it into the stacks'''
        for instruction in genome:
            if type(instruction) == int:
                self.int_stack.append(instruction)
            elif type(instruction) == float:
                self.float_stack.append(instruction)
            elif type(instruction) == bool:
                self.bool_stack.append(instruction)
            elif type(instruction) == str:
                if instruction in self.instructions_map:
                    self.exec_stack.append(self.instructions_map[instruction])
                else:
                    self.str_stack.append(instruction)
            elif type(instruction) == torch.Tensor:
                self.tensor_stack.append(instruction)

    def run(self):
        '''Runs the interpreter'''
        while self.exec_stack:
            instruction = self.exec_stack.pop(0)
            if callable(instruction):
                instruction()
            else:
                raise ValueError(f'Invalid instruction: {instruction}')

