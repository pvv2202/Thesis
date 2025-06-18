# Thesis

## Overview

This project explores the use of Genetic Programming (GP) for evolving Neural Network (NN) architectures. The primary goal is to automate the design of NNs, potentially discovering novel and efficient architectures for various machine learning tasks.

The core components of this project include:

*   **PyTorch GP:** Implementations of GP algorithms using PyTorch for NN representation and evolution.
*   **TensorFlow GP:** Similar GP implementations, but utilizing TensorFlow as the backend for NN operations.
*   **Datasets:** A collection of datasets used for training and evaluating the evolved NNs.

All train and test data should currently be in the form of a PyTorch DataLoader.

## PyTorch-based GP System

This section details the components of the Genetic Programming system implemented using PyTorch.

### Core Files and Their Roles:

*   **`gp.py`**: This file is central to the genetic programming process.
    *   **`Genome`**: Represents an individual in the population. It holds the sequence of instructions (genes) that define a neural network architecture. It includes methods for initialization, mutation (e.g., `umad` - Uniform Mutation by Addition and Deletion), and transcription into a network.
    *   **`Population`**: Manages a collection of `Genome` objects. It handles the evolutionary process, including selection (e.g., tournament selection, epsilon-lexicase), generating new generations, and running the overall evolutionary loop. It orchestrates the training and evaluation of individual genomes.

*   **`network.py`**: Defines the structure and functionality of the neural networks evolved by the GP system.
    *   **`Network`**: This class takes a network definition (a Directed Acyclic Graph or DAG) from the `Interpreter` and constructs a PyTorch `nn.Module`. It implements the `forward` pass, training (`fit`), and evaluation (`evaluate`) methods for the neural network. It also calculates parameter counts and FLOPs.

*   **`interpreter.py`**: Responsible for translating a `Genome`'s instruction sequence into a neural network structure.
    *   **`Interpreter`**: This class reads a genome (a list of instructions and values) and executes these instructions sequentially. It uses various stacks (`int`, `sint`, `float`, `bool`, `exec`, etc.) to manage operations and operands. As instructions are processed, it constructs a `DAG` representing the neural network, which is then passed to the `Network` class. It handles adding input, output, and potentially embedding layers.

*   **`instructions.py`**: Contains the set of operations that can be included in a `Genome`.
    *   **`Instructions`**: This class defines all possible operations (e.g., `matmul`, `conv2d`, `relu`, `mat_add`, `flatten`, stack operations like `dup`, `identity`) that the `Interpreter` can execute. Each instruction is a static method that manipulates the `Interpreter`'s stacks and/or modifies the `DAG` being built. The available instructions can be customized (e.g., by excluding certain activation functions).

*   **`dag.py`**: Provides the data structure for representing neural networks internally before they are converted into PyTorch models.
    *   **`Node`**: Represents a single operation or layer in the neural network. It stores its output shape, the function it performs (`fn`), a description, and its layer in the graph.
    *   **`DAG`**: A Directed Acyclic Graph class that stores `Node` objects and the connections (edges) between them. It provides methods for adding edges, getting parent nodes, and pruning unnecessary nodes from the graph (e.g., those not contributing to the final output or a recurrent connection).

### High-Level Overview of Evolution and Training:

1.  **Initialization**: A `Population` of `Genome` objects is created. Each `Genome` is initialized with a random sequence of instructions and values from `instructions.py`.
2.  **Transcription & Evaluation (per Genome)**:
    *   The `Interpreter` takes a `Genome`'s instruction list.
    *   It processes the instructions one by one, using stacks to manage operands and building a `DAG` in `dag.py` that represents the neural network architecture.
    *   This `DAG` is then used by the `Network` class in `network.py` to create an actual PyTorch `nn.Module`.
    *   The `Network` is trained on a training dataset and evaluated on a test dataset. Its fitness (e.g., accuracy, loss, parameter count) is recorded in its `Genome` object.
3.  **Selection**: Based on the fitness scores, `Genome` objects are selected from the current population to become parents for the next generation. Selection methods like tournament selection or epsilon-lexicase are implemented in `Population`.
4.  **Reproduction/Variation**:
    *   Selected `Genome` objects are copied.
    *   Mutation operations (defined in `Genome`, like `umad`) are applied to the copies to introduce variations in their instruction sequences.
5.  **New Generation**: The mutated genomes form the new population.
6.  **Iteration**: Steps 2-5 are repeated for a specified number of generations. The system aims to evolve genomes that produce high-performing neural networks.
7.  **Output**: Throughout the process, the system can save the best-performing genomes or populations. After the run, statistics like accuracy, loss, and genome size over generations are often plotted.

## TensorFlow-based GP System

This section outlines the components of the Genetic Programming system implemented using TensorFlow/Keras. This system appears to be a more direct GP approach where the genome itself is a list of Keras layer construction and training commands.

### Core Files and Their Roles:

*   **`TF/tf_gp.py`**: This file contains the core logic for the genetic programming process using TensorFlow.
    *   **`Genome`**: Represents an individual neural network. The `genes` attribute is a list of strings and values that directly map to Keras model construction steps (e.g., layer types, activation functions, parameters) and training commands. It includes methods for adding, removing, and mutating these genes. The `transcribe` method uses the `TFInterpreter` to build and evaluate the Keras model.
    *   **`Population`**: Manages a group of `Genome` objects. It handles initializing the population with random genomes, moving to the next generation (which involves sorting by fitness, cloning top performers, and mutating others), and running the main evolutionary loop.

*   **`TF/tf_interpreter.py`**: This file is responsible for interpreting the gene sequence from a `Genome` and constructing/training a Keras model.
    *   **`TFInterpreter`**: This class takes a genome (list of genes) and sequentially processes it. It maintains stacks for integers, floats, booleans, and strings, which are popped and used as parameters for Keras layers or training procedures. It has methods corresponding to Keras operations like `dense`, `conv`, `dropout`, `compile`, `fit`, and `evaluate`. The `run` method executes the gene sequence to build, train, and evaluate the Keras model, returning its performance.
    *   **Instruction/Activation/Parameter Constants**: The file also defines lists of valid instructions (e.g., `valid_instruction_mutations`), activations (`valid_activation_mutations`), loss functions, optimizers, and metrics that can be part of a genome. Default values for parameters like `DIM`, `CHANNELS`, `KERNEL`, etc., are also defined.

### High-Level Overview of Evolution and Training:

1.  **Initialization**: A `Population` of `Genome` objects is created. Each `Genome` is initialized with a base set of genes (`input_layer`, `output_layer`, `compile`, `fit`, `evaluate`) and then a number of random genes (layer definitions, parameters, activation functions) are added.
2.  **Transcription & Evaluation (per Genome)**:
    *   For each `Genome`, a `TFInterpreter` instance is created.
    *   The `TFInterpreter` reads the `Genome`'s gene list, populating its internal stacks.
    *   The `run` method of the interpreter is called. This method sequentially pops instructions and parameters from the stacks and uses them to build a Keras `Sequential` model by adding layers (e.g., `Dense`, `Conv2D`, `Dropout`).
    *   Once the model structure is defined, instructions like `compile`, `fit`, and `evaluate` are executed on the Keras model using the provided training and testing data.
    *   The evaluation score (e.g., loss or accuracy) is stored as the `Genome`'s fitness.
3.  **Selection and Reproduction**:
    *   Genomes in the `Population` are sorted by their fitness.
    *   A simple selection strategy is used: the bottom-performing genomes are removed, and the top-performers are cloned to replace them.
    *   The remaining (and newly cloned) genomes undergo mutation: genes are randomly added, removed, or their values/types are changed.
4.  **New Generation**: The modified genomes form the new population.
5.  **Iteration**: Steps 2-4 are repeated for a specified number of generations. The system aims to evolve genomes that define high-performing Keras models.
6.  **Output**: The fitness and genome of the best individual in each generation are typically printed.

## Experimental GP System (`Random/random_gp.py`)

This section describes an alternative or experimental Genetic Programming system found in `Random/random_gp.py`. Its structure suggests a different approach to genome representation and evolution compared to the PyTorch and TensorFlow systems.

### Core File and Its Role:

*   **`Random/random_gp.py`**: This file contains an independent GP implementation.
    *   **`Genome`**: Represents an individual program. Its `genome` is a list of mixed-type genes, including integers, floats, instruction strings (from `random_instructions.py`), and randomly generated PyTorch tensors.
        *   It has methods for random initialization (`initialize_random`), adding random genes (`add_gene`), removing genes (`remove_gene`), and mutating existing genes (`mutate`).
        *   The `evolve` method randomly chooses between adding, removing, or mutating a gene.
        *   The `transcribe` method uses an `Interpreter` (from `random_interpreter.py`) to process the genome and create a network.
    *   **`Population`**: Manages a collection of these `Genome` objects.
        *   It initializes a population of specified `size` with genomes having `num_initial_genes`.
        *   The `forward_generation` method implements a simple evolutionary strategy: it sorts the population by fitness (lower is better, as fitness seems to represent loss), copies the top 3 performing genomes to replace the bottom 3, and then applies the `evolve` method to these new copies.
        *   The `run` method iterates for a number of `generations`, transcribing each genome into a network, training it (the `fit` method of the network seems to return the fitness), and then advancing the generation.

### System Characteristics and Usage:

This system appears to be more experimental:

*   **Genome Structure**: The genome is a heterogeneous list of raw values, PyTorch tensors, and instruction strings. This differs from the more structured instruction lists in the PyTorch GP system or the Keras command sequence in the TensorFlow GP system.
*   **Interpreter**: It relies on `random_interpreter.py` and `random_instructions.py` (not detailed here) to make sense of its unique genome structure and build a network.
*   **Evolution Strategy**: The evolutionary strategy is basic, involving sorting, direct replacement of the worst by copies of the best, and random evolution of these copies.
*   **Tensor Generation**: Genes can be entire PyTorch tensors generated with random dimensions.

**Note**: The specific mechanisms of how `random_interpreter.py` processes these genomes and how networks are constructed and trained are not covered in this overview. Users interested in this particular GP variant will need to investigate `Random/random_interpreter.py` and `Random/random_instructions.py` for a complete understanding of its functionality and intended use case. The comments within `random_gp.py` (e.g., "TODO: Look into allometric growth...") suggest it might be a testbed for more biologically inspired or robust variation mechanisms.

## Datasets

The `Datasets/` directory contains Python scripts used for loading and preprocessing various datasets for training and evaluating the evolved neural networks. The primary approach seems to be to prepare data in the form of PyTorch DataLoaders, as mentioned in the Overview.

Available dataset scripts include:

*   `alice.py`
*   `arc_agi.py`
*   `cifar10.py`
*   `fashion_mnist.py`
*   `mnist.py`
*   `penn_treebank.py`
*   `shakespeare.py`
*   `tiny_shakespeare.py`

These scripts likely handle downloading, transforming, and wrapping datasets into a format consumable by the GP systems (primarily PyTorch DataLoaders). For specific details on how each dataset is handled, its source, and any particular preprocessing steps, please refer to the individual Python scripts within the `Datasets/` directory.

To use a new dataset with this project, you would typically:
1.  Create a new Python script in the `Datasets/` directory.
2.  In this script, implement the necessary logic to download, load, preprocess, and (if applicable) convert your dataset into PyTorch DataLoaders for training and testing.
3.  Import and use this script within your main GP training pipeline.

## Testing

The `Tests/` directory contains scripts for testing various components of the project.

Available test scripts include:

*   **`gpu_test.py`**: Likely used to verify GPU availability and basic CUDA functionality, which is crucial for training neural networks efficiently.
*   **`pytorch_net_tests.py`**: This script probably contains unit tests or integration tests for components of the PyTorch-based GP system, such as network creation, layer operations, or specific functionalities within `network.py` or `interpreter.py`.
*   **`simple_network_test.py`**: Suggests a basic test for ensuring that a simple, predefined network can be created, trained, or evaluated, possibly serving as a sanity check for the core training loop or network evaluation.

To understand the specifics of what each test script does, its dependencies, and how to run it, please refer to the content of the individual scripts within the `Tests/` directory. You would typically run these scripts from the command line (e.g., `python Tests/gpu_test.py`).

## Getting Started

This section provides a general guide to setting up and running experiments with this project.

### Dependencies

*   **Python**: This project requires Python. Ensure you have a Python interpreter installed.
*   **Core Libraries**: Based on the imports observed in the various Python scripts, common dependencies include:
    *   `torch` (PyTorch)
    *   `tensorflow` (TensorFlow/Keras)
    *   `numpy`
    *   `matplotlib`
    *   You can typically install these using pip:
        ```bash
        pip install torch tensorflow numpy matplotlib
        ```
*   **Other Dependencies**: Some scripts, particularly within the `Datasets/` or `Tests/` directories, might have additional specific dependencies (e.g., `pygame`, `tqdm`, `keras` if not using `tensorflow.keras`). Always check the import statements at the beginning of a script if you encounter import errors.