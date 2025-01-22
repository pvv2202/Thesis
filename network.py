import torch
from collections import deque
from line_profiler import profile
from tqdm import tqdm

class Network:
    def __init__(self, dag, train, test, params, device):
        '''Initialize network object'''
        self.dag = dag
        self.train = train
        self.test = test
        self.params = params
        self.device = device
        self.param_count = sum(p.numel() for p in self.params) # Number of elements across all parameter arrays
        self.flops = self.calculate_flops()

    def calculate_flops(self):
        '''Calculate and return flops'''
        flops = 0

        #BFS
        queue = deque()
        queue.extend(self.dag.graph[self.dag.root])
        while queue:
            node = queue.popleft()
            flops += node.flops
            queue.extend(self.dag.graph[node]) # Add children to queue

        return flops

    @profile
    def forward(self, x):
        '''Forward pass through the graph'''
        # Load tensor into initial node
        self.dag.root.tensor = x
        out = self.dag.root
        # BFS
        queue = deque()
        queue.extend(self.dag.graph[self.dag.root]) # Initialize to be root has no function
        while queue:
            node = queue.popleft()
            queue.extend(self.dag.graph[node]) # Add children to queue
            node.execute(self.params, self.device) # Execute the function at node
            out = node
            # print(f"Node: {node.desc}, Expected Shape: {node.shape}, Actual Shape: {node.tensor.shape}")

        if len(out.tensor.shape) > 2:
            print("Output too big")
            return None

        return out.tensor # Output node will always come last since we step layer by layer and prune

    def loss(self, y_pred, y, loss=torch.nn.functional.cross_entropy):
        '''Calculate loss'''
        # print(f"Shape of y_pred: {y_pred.shape}")
        # print(f"Shape of y: {y.shape}")
        if y_pred is None:
            print("Invalid network")
            print(self.dag)
            return float('-inf')

        return loss(y_pred, y)

    def fit(self, epochs=3, learning_rate=0.01, loss_fn=torch.nn.functional.cross_entropy, optimizer_class=torch.optim.Adam):
        '''Fit the model'''
        optimizer = optimizer_class(self.params, lr=learning_rate)

        for epoch in range(epochs):
            # Iterate over the training data, use tqdm to show a progress bar
            for x, y in tqdm(self.train, desc=f"Epoch {epoch + 1}/{epochs}", unit='batch', colour='green'):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.forward(x)

                # Forward pass and compute loss
                l = self.loss(y_pred, y, loss_fn)

                # If the network is invalid, return
                if isinstance(l, float):
                    return

                # Backpropagation
                l.backward()

                # Update parameters
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs} finished.")

    def evaluate(self, loss_fn=torch.nn.functional.cross_entropy):
        '''Evaluate the model on the test set. Returns loss, accuracy tuple'''
        correct_predictions = 0
        total_loss = 0
        test_sum = 0

        results = {}

        for i, (x, y) in enumerate(self.test):
            x, y = x.to(self.device), y.to(self.device)
            # Forward pass and compute
            with torch.no_grad():
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

                test_sum += len(y) # Should be a batch of labels

                # Calculate accuracy
                _, predictions = torch.max(y_pred, -1) # Should always be last dimension (hence -1). Max, indices is the form

                correct_predictions += (predictions == y).sum().item()
                results[i] = (total_loss, correct_predictions / len(y)) # Store the total loss and accuracy for each batch

        # Calculate accuracy
        accuracy = correct_predictions / test_sum
        print(f'Test set: Loss: {total_loss / len(self.test)}, Accuracy: {accuracy * 100:.2f}%')

        return total_loss, accuracy, results

    def __str__(self):
        return self.dag.__str__() + '\n' + f"Parameters: {self.param_count}" + '\n' + f"FLOPs: {self.flops}"