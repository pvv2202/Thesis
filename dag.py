import heapq
from collections import defaultdict, deque

class Node:
    """Node in a Directed, Acyclic Graph"""
    def __init__(self, shape, layer, fn=None, desc=None, flops=0):
        self.shape = shape # Output shape of this node after functions are applied
        self.layer = layer
        self.fn = fn
        self.desc = desc
        self.flops = flops

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

    def get_parents(self):
        """Return a dictionary of node: list of parents"""
        parents = defaultdict(list) # Everything has an empty list as a parent to start
        for node, children in self.graph.items():
            for c in children:
                parents[c].append(node) # Just add node as a parent of c
        return dict(parents)

    def prune(self, node, recurrences):
        """Prune all node that are not in the path from the root to node"""
        visited = set()
        recurrences = set(recurrences.values()) # Very minor speedup
        max_heap = []
        counter = 0 # To break ties when layers are equal

        heapq.heappush(max_heap, (-node.layer, counter, node))
        parents = self.get_parents()
        counter += 1

        # Max-Heap BFS from node to root so we explore by layer. Otherwise, we can have loop infinitely.
        while max_heap:
            _, _, current = heapq.heappop(max_heap)  # Extract node with the highest layer (breaking ties by counter)
            if current not in visited:
                visited.add(current)
                if current in recurrences:
                    visited.add(current)
                    del recurrences[current]  # Remove from recurrences if we find it in the path
                if current in parents and len(parents[current]) > 0: # If current node has parents
                    for parent in parents[current]:
                        if parent not in visited:
                            heapq.heappush(max_heap, (-parent.layer, counter, parent))  # Tie-breaking with counter
                            counter += 1

        # Reverse BFS from all recurrent nodes remaining (weren't seen in original BFS)
        for recurrent in recurrences:
            max_heap = []
            counter = 0  # To break ties when layers are equal
            heapq.heappush(max_heap, (-recurrent.layer, counter, recurrent))
            _, _, current = heapq.heappop(max_heap)  # Extract node with the highest layer (breaking ties by counter)
            if current not in visited:
                visited.add(current)
                if current in parents and len(parents[current]) > 0: # If current node has parents
                    for parent in parents[current]:
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
        # TODO: Fix this representation`
        heap = [] # Min heap (by layer) for the children of the root
        result = [] # List of strings to store the result
        layer_content = [] # Current layer content (to print by layer)
        count = 0  # Tie-breaker counter. Otherwise, we get an error trying to compare nodes
        curr_layer = 0
        seen = set()
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
                if child in seen:
                    continue
                heapq.heappush(heap, (child.layer, count, child))
                count += 1
                seen.add(child)

        # Append any remaining content from the last layer
        if layer_content:
            result.append(" ".join(layer_content))

        return "\n".join(result)

# We read the genome, then we step through to create the DAG. Function to create the DAG. To create, we take the first tensor from the stack
# And make that a node with layer = 0 (this is our input layer).