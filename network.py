import torch
from collections import deque
from line_profiler import profile
from tqdm import tqdm
import heapq
import networkx as nx
import matplotlib.pyplot as plt

class Network:
    def __init__(self, dag, params, device, recurrent=False):
        '''Initialize network object'''
        self.dag = dag
        self.params = params
        self.device = device
        self.param_count = sum(p.numel() for p in self.params) # Number of elements across all parameter arrays
        self.flops = self.calculate_flops()
        self.recurrent = recurrent

    def calculate_flops(self):
        '''Calculate and return flops'''
        flops = 0

        # BFS
        queue = deque()
        queue.extend(self.dag.graph[self.dag.root])
        while queue:
            node = queue.popleft()
            flops += node.flops
            queue.extend(self.dag.graph[node]) # Add children to queue

        return flops

    @profile
    def forward(self, x, h=None):
        '''Forward pass through the graph'''
        # Load tensor into the root node
        self.dag.root.tensor = x
        out = self.dag.root

        # Update hidden node if recurrent
        if self.recurrent:
            self.dag.hidden_node.tensor = h

        # Min heap (by layer) for the children of the root
        heap = []
        count = 0  # Tie-breaker counter. Otherwise we get an error trying to compare nodes
        for child in self.dag.graph[self.dag.root]:
            # Push a tuple of (layer, counter, node) into the heap
            heapq.heappush(heap, (child.layer, count, child))
            count += 1

        # Pop nodes in order of increasing layer
        while heap:
            _, _, node = heapq.heappop(heap)
            for child in self.dag.graph[node]:
                heapq.heappush(heap, (child.layer, count, child))
                count += 1

            # Execute node
            node.execute(self.params, self.device)
            out = node

        return out.tensor  # Output is always the last processed node

    def loss(self, y_pred, y, loss_fn=torch.nn.functional.cross_entropy):
        '''Calculate Loss'''
        if y_pred is None:
            print("Invalid network")
            print(self.dag)
            return float('-inf')

        # Flatten for cross entropy
        # (batch_size * seq_len, vocab_size) vs. (batch_size * seq_len)
        y_pred_flat = y_pred.view(-1, y_pred.size(-1))  # (N, C)
        y_flat = y.view(-1)  # (N,)

        return loss_fn(y_pred_flat, y_flat)

    def fit(self, train, test, epochs=3, learning_rate=0.001, loss_fn=torch.nn.functional.cross_entropy, optimizer_class=torch.optim.Adam, drought=False, generation=None, downsample=None):
        '''Fit the model'''
        if train == None:
            print("No training data provided")
            return

        optimizer = optimizer_class(self.params, lr=learning_rate)
        train_fraction = 1
        if generation:
            train_fraction = epochs*generation # Get fraction of data we want to train with

        for epoch in range(epochs):
            # Iterate over the training data, use tqdm to show a progress bar
            progress_bar = tqdm(train, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", colour="green")

            # If recurrent, reset hidden node
            if self.recurrent:
                x, y = next(iter(train))
                batch_dim = x.shape[0]
                prev_y = torch.zeros(batch_dim, *self.dag.hidden_node.shape).to(self.device)

            for i, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if self.recurrent:
                    y_pred = self.forward(x=x, h=prev_y)
                    prev_y = y_pred.detach() # Detach to allow for backpropagation
                else:
                    y_pred = self.forward(x)

                # Forward pass and compute loss
                l = self.loss(y_pred, y, loss_fn)

                # Backpropagation
                l.backward()

                # Update parameters
                optimizer.step()

                if downsample is not None and (i + 1) / len(train) >= downsample:
                    return None

                # Iteratively increase the amount we train
                if generation:
                    fraction_done = (i + 1) / len(train)
                    if fraction_done >= train_fraction:
                        return None

                # If drought is on and we've trained on 25%, test to see if we stop here
                # TODO: Hard-coded threshold for now
                if drought and i == len(train) // 4:
                    loss, accuracy, results = self.evaluate(test)
                    if accuracy <= 0.15:
                        return (loss, accuracy, results)

        return None

    def evaluate(self, test, loss_fn=torch.nn.functional.cross_entropy):
        # TODO: Update to handle reccurent networks
        '''Evaluate the model on the test set. Returns loss, accuracy tuple'''
        if test == None:
            print("No testing data provided")
            return

        correct_predictions = 0
        total_loss = 0
        test_sum = 0

        results = {}

        # If recurrent, reset hidden node
        if self.recurrent:
            x, y = next(iter(test))
            batch_dim = x.shape[0]
            prev_y = torch.zeros(batch_dim, *self.dag.hidden_node.shape).to(self.device)

        for x, y in test:
            x, y = x.to(self.device), y.to(self.device)
            # Forward pass and compute
            with torch.no_grad():
                if self.recurrent:
                    y_pred = self.forward(x=x, h=prev_y)
                    prev_y = y_pred.detach() # Detach to allow for backpropagation
                else:
                    y_pred = self.forward(x)

                # print("Prediction:")
                # print(torch.max(y_pred, -1))
                # print("Actual:")
                # print(y)

                l = self.loss(y_pred, y, loss_fn)
                # If invalid, just return inf, -inf
                if type(l) == float:
                    return float('inf'), float('-inf')
                total_loss += l.item()

                test_sum += y.numel() # Number of elements in y

                # Calculate accuracy
                _, predictions = torch.max(y_pred, -1) # Should always be last dimension (hence -1). Max, indices is the form

                correct_predictions += (predictions == y).sum().item()
                results[x] = (total_loss, correct_predictions / len(y)) # Store the total loss and accuracy for each batch

            # if i/len(self.test) >= 0.25:
            #     break

        # Calculate accuracy
        accuracy = correct_predictions / test_sum
        print(f'Test set: Loss: {total_loss / len(test)}, Accuracy: {accuracy * 100:.2f}%')

        return total_loss, accuracy, results

    def visualize(self):
        # Create the graph
        G = nx.DiGraph()  # Use DiGraph for directed edges
        for node in self.dag.graph.keys():
            G.add_node((node.desc, node.shape), layer=node.layer)

        directed_edges = []
        parents = []
        for node, edges in self.dag.graph.items():
            for edge in edges:
                directed_edges.append(((node.desc, node.shape), (edge.desc, edge.shape)))
            if node.parents is not None:
                for parent in node.parents:
                    parents.append(((node.desc, node.shape), (parent.desc, parent.shape)))

        # Add edges to the graph
        G.add_edges_from(directed_edges)
        G.add_edges_from(parents)

        pos = nx.multipartite_layout(G, subset_key="layer")

        plt.figure(figsize=(len(self.dag.graph.keys()), len(self.dag.graph.keys())))
        nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=5000, edge_color="black", font_size=10)
        nx.draw_networkx_edges(G, pos, edgelist=directed_edges, edge_color="black", width=3, style="solid")
        nx.draw_networkx_edges(G, pos, edgelist=parents, edge_color="red", width=1, style="dashed")

        plt.title("Layered Computation Graph")
        plt.show()

    def __str__(self):
        return self.dag.__str__() + '\n' + f"Parameters: {self.param_count}" + '\n' + f"FLOPs: {self.flops}"

