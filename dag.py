import torch
import heapq
from collections import deque

class Node:
    """Node in a Directed, Acyclic Graph"""
    def __init__(self, shape, layer, fn=None, parents=None, weight_id=None, desc=None, flops=0):
        self.shape = shape # Output shape of this node after functions are applied
        self.layer = layer
        self.fn = fn
        self.parents = parents
        self.weight_id = weight_id
        self.desc = desc
        self.tensor = None
        self.flops = flops

    def execute(self, params, device):
        """Execute the function on the input, store the result"""
        # Do nothing if no function or parents have no tensor
        if self.fn is None or any(parent.tensor is None for parent in self.parents):
            return self.tensor

        parent_tensors = [parent.tensor for parent in self.parents]

        # Otherwise, execute the function
        if self.weight_id is None:
            self.tensor = self.fn(*parent_tensors)
        else:
            self.tensor = self.fn(*parent_tensors, params[self.weight_id])

        # Send to device
        self.tensor = self.tensor.to(device)

        return self.tensor

class DAG:
    """Directed, Acyclic Graph"""
    def __init__(self, root):
        self.graph = {}

        # Add root node to graph
        self.root = root
        self.graph[root] = []

    def add_edge(self, u, v):
        """Add an edge from u to v"""
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []

        self.graph[u].append(v)

    def remove_edge(self, u, v):
        """Remove an edge from u to v"""
        self.graph[u].remove(v)

    def prune(self, node):
        """Prune all node that are not in the path from the root to node"""
        visited = set()
        max_heap = []
        counter = 0 # To break ties when layers are equal

        heapq.heappush(max_heap, (-node.layer, counter, node))
        counter += 1

        # Max-Heap BFS from node to root so we explore by layer. Otherwise, we can have loop infinitely.
        while max_heap:
            _, _, current = heapq.heappop(max_heap)  # Extract node with the highest layer (breaking ties by counter)
            if current not in visited:
                visited.add(current)
                if current.parents:
                    for parent in current.parents:
                        if parent not in visited:
                            heapq.heappush(max_heap, (-parent.layer, counter, parent))  # Tie-breaking with counter
                            counter += 1

        for u in list(self.graph.keys()):
            if u not in visited:  # If node isn't in the path from root to node, remove it
                del self.graph[u]
            else:  # Otherwise, remove any edges to nodes that are not in the path
                self.graph[u] = [v for v in self.graph[u] if v in visited]

    def __str__(self):
        """String representation of the DAG"""
        heap = [] # Min heap (by layer) for the children of the root
        result = [] # List of strings to store the result
        layer_content = [] # Current layer content (to print by layer)
        count = 0  # Tie-breaker counter. Otherwise, we get an error trying to compare nodes
        curr_layer = 0
        for child in self.graph[self.root]:
            # Push a tuple of (layer, counter, node) into the heap
            heapq.heappush(heap, (child.layer, count, child))
            count += 1

        # Pop nodes in order of increasing layer
        while heap:
            _, _, node = heapq.heappop(heap)
            if node.layer > curr_layer:
                result.append(" ".join(layer_content))  # Join all nodes in the current layer
                layer_content = []
                curr_layer = node.layer

            layer_content.append(f'Layer {node.layer}: {node.shape}; Fn: {node.desc}')

            for child in self.graph[node]:
                heapq.heappush(heap, (child.layer, count, child))
                count += 1

        # Append any remaining content from the last layer
        if layer_content:
            result.append(" ".join(layer_content))

        return "\n".join(result)

# We read the genome, then we step through to create the DAG. Function to create the DAG. To create, we take the first tensor from the stack
# And make that a node with layer = 0 (this is our input layer).