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
        }

        """Dense layer would be like 1x5 x 5x4 to get 1x4. Attention is obviously all matrix mult.
        Conv layer is more complicated. Maybe need to define "complete mult" function or something that convolves. 
        Also need to figure out how to implement reshaping and padding and stuff. 
        """

    '''
    Tensor Operations
    '''

    def tensor_matmul(self):
        '''Performs a matrix multiplication on the top two tensors on the stack'''
        if len(self.tensor_stack) >= 2:
            a = self.tensor_stack.pop()
            b = self.tensor_stack.pop()

            # Check if the dimensions are compatible. If not, noop
            if a.size()[-1] != b.size()[-2]:
                return

            result = torch.matmul(a, b)
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

