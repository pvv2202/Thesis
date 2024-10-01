import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import Interpreter

if __name__ == '__main__':
    # Load data (example with the MNIST dataset)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the dataset to be between 0 and 1 (optional but recommended for neural networks)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Reshape data
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # Convert labels to one-hot encoded
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Initialize the PushInterpreter
    interpreter = Interpreter.PushInterpreter(X_train, y_train, X_test, y_test)

    # Add code
    interpreter.exec_stack.extend([
        interpreter.input_layer,
        interpreter.conv,
        interpreter.conv,
        interpreter.max_pool,
        interpreter.normalize,
        interpreter.conv,
        interpreter.max_pool,
        interpreter.normalize,
        interpreter.global_pool,
        interpreter.output_layer,
        interpreter.compile,
        interpreter.fit
    ])

    # Add ints
    interpreter.int_stack.extend([
    ])

    # Add strings
    interpreter.str_stack.extend([
        'relu',
        'relu',
        'relu',
        'relu',
    ])

    interpreter.run()

