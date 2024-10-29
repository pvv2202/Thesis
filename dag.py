class Node:
    '''Node in a Directed, Acyclic Graph'''
    def __init__(self, shape, layer, fn, parents, weights=None):
        self.shape = shape # Output shape of this node after functions are applied
        self.layer = layer
        self.fn = fn
        self.parents = parents
        self.weights = weights

    def execute(self):
        '''Execute the function on the input'''
        if self.weights is None:
            return self.fn(self.parents)
        else:
            return self.fn(self.parents, self.weights)

class DAG:
    '''Directed, Acyclic Graph'''
    def __init__(self, root):
        self.graph = {}

        # Add root node to graph
        self.root = root
        self.graph[root] = []

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []

        self.graph[u].append(v)

    def remove_edge(self, u, v):
        self.graph[u].remove(v)

    def __str__(self):
        return str(self.graph)


# We read the genome, then we step through to create the DAG. Function to create the DAG. To create, we take the first tensor from the stack
# And make that a node with layer = 0 (this is our input layer).