import keras
from keras import layers

class PushInterpreter():
    '''Interprets and processed Push programs'''
    DIM = 32
    KERNEL = 3
    DROPOUT = 0.5
    ACTIVATION = 'relu'
    OPTIMIZER = 'adam'
    LOSS = 'categorical_crossentropy'
    METRICS = 'accuracy'

    valid_instructions = {'dense', 'conv', 'dropout', 'normalize', 'global_pool', 'max_pool', 'compile', 'input_layer', 'output_layer', 'fit'} # Set of all valid code types
    valid_activations = {'relu', 'sigmoid', 'softmax'} # Set of all valid activations
    valid_loss = {'sparse_categorical_crossentropy', 'binary_crossentropy', 'categorical_crossentropy'} # Set of all valid losses
    valid_optimizers = {'adam', 'sgd', 'rmsprop'} # Set of all valid optimizers
    valid_metrics = {'accuracy', 'precision', 'recall'} # Set of all valid metrics

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
            'fit': self.fit
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
                return self.ACTIVATION
            elif type == 'optimizer':
                return self.OPTIMIZER
            elif type == 'loss':
                return self.LOSS
            elif type == 'metrics':
                return self.METRICS
            else:
                return ''
        else:
            string = self.str_stack.pop()
            if type == 'activation' and string not in self.valid_activations:
                return self.ACTIVATION
            elif type == 'optimizer' and string not in self.valid_optimizers:
                return self.OPTIMIZER
            elif type == 'loss' and string not in self.valid_loss:
                return self.LOSS
            elif type == 'metrics' and string not in self.valid_metrics:
                return self.METRICS
            else:
                return string

    """STACK OPERATIONS"""
    def pop_int(self, type=None):
        '''Pops an int from the stack'''
        if len(self.int_stack) == 0:
            if type == 'dim':
                return self.DIM
            elif type == 'kernel':
                return self.KERNEL
            else:
                return 0
        return self.int_stack.pop()

    def pop_float(self, type=None):
        '''Pops a float from the stack'''
        if len(self.float_stack) == 0:
            if type == 'dropout':
                return self.DROPOUT
            else:
                return 0.0
        return self.float_stack.pop()

    """TENSORFLOW FUNCTIONS"""
    def dense(self):
        '''Adds a dense layer to the model'''
        dim = self.pop_int('dim')
        activation = self.pop_string('activation')

        self.model.add(layers.Dense(dim, activation=activation))

    def conv(self):
        '''Adds a convolutional layer to the model'''
        dim = self.pop_int('dim')
        activation = self.pop_string('activation')
        k = self.pop_int('kernel')

        self.model.add(layers.Conv2D(dim, kernel_size=(k, k), activation=activation))

    def dropout(self):
        '''Adds a dropout layer to the model'''
        rate = self.pop_float('dropout')

        self.model.add(layers.Dropout(rate=rate))

    def normalize(self):
        '''Adds a normalization layer to the model'''
        self.model.add(layers.BatchNormalization())

    def global_pool(self):
        '''Adds a global pooling layer to the model'''
        self.model.add(layers.GlobalAveragePooling2D())

    def max_pool(self):
        '''Adds a max pooling layer to the model'''
        self.model.add(layers.MaxPool2D())

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
        self.model.add(keras.Input(shape=(28, 28, 1)))

    def output_layer(self):
        '''Adds an output layer to the model. This should be predefined'''
        self.dense()
        self.model.add(keras.layers.Dense(10, activation='softmax'))

    def fit(self):
        '''Fits the model to the data'''
        self.model.fit(self.X_train, self.y_train, epochs=5)

    def evaluate(self):
        '''Evaluates the model on the test data'''
        self.model.evaluate(self.X_test, self.y_test)

    def run(self):
        '''Runs the interpreter'''
        while self.exec_stack:
            instruction = self.exec_stack.pop(0)
            print(instruction)
            if callable(instruction):
                instruction()
            else:
                raise ValueError(f'Invalid instruction: {instruction}')