import keras
from keras import layers

#TODO: ADD STRIDE ABILITY, IMPLEMENT GLOBAL POOLING IN A WAY THAT IT DOESN'T JUST BREAK EVERY MODEL THAT HAS IT

DIM = 32
CHANNELS = 5
KERNEL = 3
POOL = 2
DROPOUT = 0.5
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = 'accuracy'
START_LAYER = [28, 28, 1]

valid_instructions = {'dense', 'conv', 'dropout', 'normalize', 'global_pool', 'max_pool', 'compile', 'input_layer', 'output_layer', 'fit', 'evaluate'}  # Set of all valid code types
valid_activations = {'relu', 'sigmoid', 'softmax'}  # Set of all valid activations
valid_loss = {'sparse_categorical_crossentropy', 'binary_crossentropy', 'categorical_crossentropy'}  # Set of all valid losses
valid_optimizers = {'adam', 'sgd', 'rmsprop'}  # Set of all valid optimizers
valid_metrics = {'accuracy', 'precision', 'recall'}  # Set of all valid metrics

valid_instruction_mutations = ['dense', 'conv', 'dropout', 'normalize', 'global_pool', 'max_pool']  # List of all valid code types
valid_activation_mutations = ['relu', 'sigmoid', 'softmax']

class PushInterpreter():
    '''Interprets and processed Push programs'''

    #TODO: Think about model fingerprint to prevent the same model from being tested. It is likely that mutations will have no change

    def __init__(self, X_train, y_train, X_test, y_test):
        # Initialize stacks
        self.int_stack = []
        self.float_stack = []
        self.bool_stack = []
        self.str_stack = []
        self.exec_stack = []

        # Initialize data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Dimensions of the output of the previous layer
        self.previous_layer = START_LAYER

        # Initialize the model
        self.model = keras.Sequential()

        # Mapping of instruction strings to method references
        self.instructions_map = {
            'dense': self.dense,
            'conv': self.conv,
            'dropout': self.dropout,
            'normalize': self.normalize,
            'global_pool': self.global_pool,
            'max_pool': self.max_pool,
            'compile': self.compile,
            'input_layer': self.input_layer,
            'output_layer': self.output_layer,
            'fit': self.fit,
            'evaluate': self.evaluate
        }

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
                # Check if instruction is a valid code type. If so, add it to exec stack
                if instruction in self.instructions_map:
                    self.exec_stack.append(self.instructions_map[instruction])
                else:
                    self.str_stack.append(instruction)

    def pop_string(self, type=None):
        '''Pops a string from the stack'''
        if len(self.str_stack) == 0:
            if type == 'activation':
                return ACTIVATION
            elif type == 'optimizer':
                return OPTIMIZER
            elif type == 'loss':
                return LOSS
            elif type == 'metrics':
                return METRICS
            else:
                return ''
        else:
            string = self.str_stack.pop()
            if type == 'activation' and string not in valid_activations:
                return ACTIVATION
            elif type == 'optimizer' and string not in valid_optimizers:
                return OPTIMIZER
            elif type == 'loss' and string not in valid_loss:
                return LOSS
            elif type == 'metrics' and string not in valid_metrics:
                return METRICS
            else:
                return string

    """STACK OPERATIONS"""
    def pop_int(self, type=None):
        '''Pops an int from the stack'''
        if len(self.int_stack) == 0:
            if type == 'dim':
                return DIM
            elif type == 'kernel':
                return KERNEL
            elif type == 'pool':
                return POOL
            elif type == 'channels':
                return CHANNELS
            else:
                return 0
        return self.int_stack.pop()

    def pop_float(self, type=None):
        '''Pops a float from the stack'''
        if len(self.float_stack) == 0:
            if type == 'dropout':
                return DROPOUT
            else:
                return 0.0
        return self.float_stack.pop()

    """HELPER FUNCTIONS"""
    def reshape_dense(self, channels):
        '''Reshapes the previous layer to a 3D tensor'''
        dim = self.pop_int('dim')
        reshape_target = (dim, dim, channels)
        # If previous layer is not the correct size, add a dense layer to reshape
        if self.previous_layer[0] != dim * dim * channels:
            activation = self.pop_string('activation')
            self.model.add(layers.Dense(dim * dim * channels, activation=activation))
        # Reshape layer into 2D tensor
        self.model.add(layers.Reshape(reshape_target))
        self.previous_layer = reshape_target

    """GENETIC FUNCTIONS"""
    def dense(self):
        '''Adds a dense layer to the model'''
        dim = self.pop_int('dim')
        activation = self.pop_string('activation')

        if len(self.previous_layer) > 1:
            self.model.add(layers.Flatten())

        self.model.add(layers.Dense(dim, activation=activation))
        self.previous_layer = [dim]

    def conv(self):
        '''Adds a convolutional layer to the model'''
        channels = self.pop_int('channels')
        activation = self.pop_string('activation')
        k = self.pop_int('kernel')
        print(k)
        print(self.previous_layer[0])

        # If previous layer is 1D, reshape it to 2D
        if len(self.previous_layer) == 1:
            self.reshape_dense(channels)

        # If k > previous dimension, do nothing
        if k > self.previous_layer[0]:
            return

        self.model.add(layers.Conv2D(channels, kernel_size=(k, k), activation=activation))

        # Calculate new dimensions, assuming stride of 1
        new_dim = self.previous_layer[0] - k + 1
        self.previous_layer = [new_dim, new_dim, channels]

    def dropout(self):
        '''Adds a dropout layer to the model'''
        rate = self.pop_float('dropout')

        self.model.add(layers.Dropout(rate=rate))

    def normalize(self):
        '''Adds a normalization layer to the model'''
        self.model.add(layers.BatchNormalization())

    def global_pool(self):
        '''Adds a global pooling layer to the model'''
        pass
        # if len(self.previous_layer) == 1:
        #     return
        #
        # self.model.add(layers.GlobalAveragePooling2D()
        #
        # self.previous_layer = ()

    def max_pool(self):
        '''Adds a max pooling layer to the model'''
        if len(self.previous_layer) == 1:
            return

        # Get pool size, add layer
        p = self.pop_int('pool')
        if p > self.previous_layer[0]:
            return

        self.model.add(layers.MaxPool2D(pool_size=(p, p)))

        # Calculate new dimensions
        new_dim = (self.previous_layer[0] - p) // p + 1
        self.previous_layer = [new_dim, new_dim, 1]

    def compile(self):
        '''Compiles the model'''
        optimizer = self.pop_string('optimizer')
        loss = self.pop_string('loss')
        metrics = self.pop_string('metrics')

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        self.model.summary()

    def input_layer(self):
        '''Adds an input layer to the model. This should be predefined'''
        self.model.add(keras.Input(shape=START_LAYER))

    def output_layer(self):
        '''Adds an output layer to the model. This should be predefined'''
        # If previous layer is multidimensional, flatten it
        if len(self.previous_layer) > 1:
            self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(10, activation='softmax'))

    def fit(self):
        '''Fits the model to the data'''
        self.model.fit(self.X_train, self.y_train, epochs=1)

    def evaluate(self):
        '''Evaluates the model on the test data'''
        score = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return score

    def run(self):
        '''Runs the interpreter'''
        while self.exec_stack:
            instruction = self.exec_stack.pop(0)
            if callable(instruction):
                if instruction == self.evaluate:
                    return instruction()
                elif instruction == self.compile:
                    instruction()
                    if self.model.count_params() > 1000000:
                        return [0]
                else:
                    instruction()

            else:
                raise ValueError(f'Invalid instruction: {instruction}')