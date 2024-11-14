import torch
from collections import deque
from line_profiler import profile

class Network:
    def __init__(self, dag, train, test, params, device):
        '''Initialize network object'''
        self.dag = dag
        self.train = train
        self.test = test
        self.params = params
        self.device = device

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
            # print(f"Expected Shape: {node.shape}")
            queue.extend(self.dag.graph[node]) # Add children to queue
            node.execute(self.params, self.device) # Execute the function at node
            # print(f"Actual Shape: {node.tensor.shape}")
            out = node
            # print(f"Intermediate Out: {out.tensor.shape}")

        # print(f"Output Shape: {out.tensor.shape}")
        return out.tensor # Output node will always come last since we step layer by layer

    def loss(self, x, y, loss=torch.nn.functional.cross_entropy):
        '''Calculate loss'''
        y_pred = self.forward(x)
        # print(f"Shape of y_pred: {y_pred.shape}")
        # print(f"Shape of y: {y.shape}")
        if y_pred.shape is None:
            return float('-inf')

        return loss(y_pred, y)

    def fit(self, epochs=3, learning_rate=0.01, loss_fn=torch.nn.functional.cross_entropy, optimizer_class=torch.optim.Adam):
        '''Fit the model'''
        optimizer = optimizer_class(self.params, lr=learning_rate)

        final_accuracy = 0
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            total_samples = 0

            for x, y in self.train:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                # Forward pass and compute loss
                l = self.loss(x, y, loss_fn)
                # Backpropagation
                l.backward()

                # for param in self.params:
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())

                # Update parameters

                # old_params = [param.clone() for param in self.params]
                optimizer.step()
                # for old_param, param in zip(old_params, self.params):
                #     print((old_param - param).abs().mean())

                # Accumulate loss
                epoch_loss += l.item()

                # Calculate accuracy
                with torch.no_grad():
                    outputs = self.forward(x)  # Forward pass
                    _, predicted = torch.max(outputs, 1)  # Get predicted class
                    correct_predictions += (predicted == y).sum().item()  # Count correct predictions
                    total_samples += y.size(0)  # Count total samples

            # Calculate epoch accuracy
            final_accuracy = epoch_accuracy = correct_predictions / total_samples
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.train)}, Accuracy: {epoch_accuracy * 100:.2f}%')

        # Return accuracy from last epoch
        return final_accuracy

    def __str__(self):
        return self.dag.__str__()