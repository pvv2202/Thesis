import torch
from collections import deque
from line_profiler import profile
from tqdm import tqdm
import heapq
import math
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Network(nn.Module):
    def __init__(self, dag, root, recurrences, device, recurrent):
        """Initialize network object"""
        super().__init__()

        self.dag = dag
        self.root = root
        self.parents = dag.get_parents()
        self.device = device
        self.recurrences = recurrences
        self.recurrent = recurrent
        self.flops = self.calculate_flops()

        # Get a linear order of nodes using topological sort
        self.order = topological_sort(self.dag.graph)

        modules = {}
        self.node_to_name = {}

        for i, node in enumerate(self.order):
            module_name = f"node_{i}"  # Assign a unique string name
            self.node_to_name[node] = module_name
            if node == root:
                modules[module_name] = nn.Identity()  # Root node is identity
            else:
                modules[module_name] = node.fn  # Use the function assigned to the node

        self.node_modules = nn.ModuleDict(modules)

        self.param_count = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """Forward pass through the network"""
        if not self.recurrent:
            # Store the output of each node. Initialize with root
            outputs = {self.root: self.node_modules[self.node_to_name[self.root]](x)}

            for node in self.order:
                if node == self.root:
                    continue

                parents = self.parents[node] # Get the parents of the node
                parent_tensors = [outputs[parent] for parent in parents] # Get the output tensors of the parents

                # Execute the function of the node
                outputs[node] = self.node_modules[self.node_to_name[node]](*parent_tensors) # Execute the function on parent tensors

            return outputs[self.order[-1]]  # Return the output of the last node
        else: # If recurrent
            outputs = []
            seq_length = x.size(1)
            for t in range(seq_length):
                x_t = x[:, t] # Get the input at this timestep
                outputs.append({self.root: self.node_modules[self.node_to_name[self.root]](x_t)})
                for node in self.order:
                    if node == self.root:
                        continue

                    parents = self.parents[node]
                    parent_tensors = [outputs[t][parent] for parent in parents]

                    # Set the parent tensors to the output of the previous source node to set its value to the previous output.
                    if node in self.recurrences and t > 0:
                        parent_tensors = [outputs[t-1][self.recurrences[node]]]

                    outputs[t][node] = self.node_modules[self.node_to_name[node]](*parent_tensors)

            return torch.stack([outputs[t][self.order[-1]] for t in range(seq_length)], dim=1) # Return the outputs at each timestep

    def fit(self, train, epochs=1, learning_rate=0.001, loss_fn=F.cross_entropy, optimizer=torch.optim.Adam, generation=None, seq_length=20):
        """Fit the model"""
        optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.to(self.device) # Move to GPU if applicable

        # Logic for increasing the amount we train as we go
        train_fraction = 1
        if generation:
            train_fraction = epochs*generation # Get fraction of data we want to train with

        for epoch in range(int(math.ceil(epochs))):
            # Progress bar
            progress_bar = tqdm(train, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            training_acc = 0
            total_predictions = 0

            for i, (x, y) in enumerate(progress_bar):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                if self.recurrent:
                    # If recurrent, unroll through time (doing truncated BPTT so steps of seq_length)
                    for t in range(0, x.size(1), seq_length):
                        # Get the input and output for this time step but preserve batch dimension
                        x_t = x[:, t:t + seq_length]
                        y_t = y[:, t:t + seq_length]
                        y_pred = self.forward(x_t)

                        # For compatibility with cross entropy, flatten the predictions and targets
                        y_pred = y_pred.view(-1, y_pred.size(-1)) # Flatten these into 2D w batch_size * seq_len, vocab_size
                        y_t = y_t.reshape(-1) # Flatten these into 1D w batch_size * seq_len

                        # Add to accuracy
                        _, predictions = torch.max(y_pred, dim=-1)
                        training_acc += (predictions == y_t).sum().item()
                        total_predictions += y_t.numel()

                        loss = loss_fn(y_pred, y_t)
                        loss.backward()
                else:
                    y_pred = self.forward(x)
                    loss = loss_fn(y_pred, y)
                    loss.backward()

                    # Add to accuracy
                    _, predictions = torch.max(y_pred, dim=-1)
                    training_acc += (predictions == y).sum().item()
                    total_predictions += y.numel()

                optimizer.step()

                progress_bar.set_postfix({"loss": float(loss.item())})

                if epoch == int(math.floor(epochs)) and int(math.floor(epochs)) + (i + 1) / len(train) >= epochs:
                    # If we're only partially training on epochs, return early
                    return

                # Iteratively increase the amount we train
                if generation:
                    fraction_done = (i + 1) / len(train)
                    if fraction_done >= train_fraction:
                        return

            print(f"\rTraining Accuracy: {training_acc / total_predictions * 100:.2f}%")

    def evaluate(self, test, loss_fn=F.cross_entropy):
        """Evaluate the model on the test set. Returns average loss, accuracy, results tuple"""
        self.eval()  # Put in eval mode (e.g., for dropout/batchnorm if needed)
        self.to(self.device)  # Move to GPU if applicable
        total_loss = 0.0
        correct_predictions = 0
        total_examples = 0
        results = []

        with torch.no_grad():
            for x, y in test:
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass
                y_pred = self.forward(x)

                # For compatibility with cross entropy, flatten the predictions and targets
                y_pred = y_pred.view(-1, y_pred.size(-1))  # Flatten these into 2D w batch_size * seq_len, vocab_size
                y = y.reshape(-1)  # Flatten these into 1D w batch_size * seq_len

                # Compute loss
                loss = loss_fn(y_pred, y)
                total_loss += loss.item()

                _, predictions = torch.max(y_pred, dim=-1)
                batch_predictions = (predictions == y).sum().item()
                correct_predictions += batch_predictions
                # Compute accuracy
                # Store the total loss and accuracy for each batch. Need this for Lexicase Selection
                results.append((loss.item(), batch_predictions / y.numel()))
                total_examples += y.numel()

        # Average loss and accuracy
        avg_loss = total_loss / len(test)
        accuracy = correct_predictions / total_examples

        print(f"Test set: Loss: {avg_loss}, Accuracy: {accuracy*100:.2f}%")
        return avg_loss, accuracy, results

    def calculate_flops(self):
        """Calculate and return flops"""
        flops = 0

        # BFS
        queue = deque()
        queue.extend(self.dag.graph[self.dag.root])
        while queue:
            node = queue.popleft()
            flops += node.flops
            queue.extend(self.dag.graph[node]) # Add children to queue

        return flops

    def __str__(self):
        return self.dag.__str__() + '\n' + f"Parameters: {self.param_count}" + '\n' + f"FLOPs: {self.flops}"


#     def visualize(self):
#         '''Visualize the network with camera movement using arrow keys or WASD.'''
#         pygame.init()
#
#         # Constants
#         NODE_RADIUS = 40
#         BACKGROUND_COLOR = (255, 255, 255)
#         NODE_COLOR = (189, 140, 245)
#         EDGE_COLOR = (0, 0, 0)
#         TEXT_COLOR = (0, 0, 0)
#         ARROW_SIZE = 10
#         WIDTH = 1200
#         HEIGHT = 800
#
#         # Display
#         screen = pygame.display.set_mode((WIDTH, HEIGHT))
#         pygame.display.set_caption("Network Visualization")
#
#         # Movement variables
#         camera_x = 0
#         camera_y = 0
#         move_speed = 10
#
#         node_positions = {self.dag.root: (50, HEIGHT // 2)}
#         heap = []
#         count = 0
#         for child in self.dag.graph[self.dag.root]:
#             heapq.heappush(heap, (child.layer, count, child))
#             count += 1
#
#         curr_layer = []
#         prev_layer = 0
#         while heap:
#             _, _, node = heapq.heappop(heap)
#             if node.layer == prev_layer:
#                 curr_layer.append(node)
#             else:
#                 for i, x in enumerate(curr_layer):
#                     node_positions[x] = (50 + prev_layer * 100, 50 + i * 50)
#                 curr_layer = [node]
#                 prev_layer = node.layer
#             curr_layer.append(node)
#             for child in self.dag.graph[node]:
#                 heapq.heappush(heap, (child.layer, count, child))
#                 count += 1
#         for i, x in enumerate(curr_layer):
#             node_positions[x] = (50 + prev_layer * 100, 50 + i * 50)
#
#         def draw_arrow(start, end, color, arrow_size=ARROW_SIZE):
#             pygame.draw.line(screen, color, start, end, 2)
#             angle = math.atan2(end[1] - start[1], end[0] - start[0])
#             x1 = end[0] - arrow_size * math.cos(angle - math.pi / 6)
#             y1 = end[1] - arrow_size * math.sin(angle - math.pi / 6)
#             x2 = end[0] - arrow_size * math.cos(angle + math.pi / 6)
#             y2 = end[1] - arrow_size * math.sin(angle + math.pi / 6)
#             pygame.draw.polygon(screen, color, [(end[0], end[1]), (x1, y1), (x2, y2)])
#
#         def draw_graph():
#             screen.fill(BACKGROUND_COLOR)
#             for node, neighbors in self.dag.graph.items():
#                 start_pos = (node_positions[node][0] + camera_x, node_positions[node][1] + camera_y)
#                 for neighbor in neighbors:
#                     if neighbor in node_positions:
#                         end_pos = (node_positions[neighbor][0] + camera_x, node_positions[neighbor][1] + camera_y)
#                         draw_arrow(start_pos, end_pos, EDGE_COLOR)
#             for node, pos in node_positions.items():
#                 adjusted_pos = (pos[0] + camera_x, pos[1] + camera_y)
#                 pygame.draw.circle(screen, NODE_COLOR, adjusted_pos, NODE_RADIUS)
#                 font = pygame.font.Font(None, 20)
#                 text_surface_desc = font.render(f"{node.desc}", True, TEXT_COLOR)
#                 text_surface_shape = font.render(f"{node.shape}", True, TEXT_COLOR)
#                 text_rect_desc = text_surface_desc.get_rect(center=(adjusted_pos[0], adjusted_pos[1] - 10))
#                 text_rect_shape = text_surface_shape.get_rect(center=(adjusted_pos[0], adjusted_pos[1] + 10))
#                 screen.blit(text_surface_desc, text_rect_desc)
#                 screen.blit(text_surface_shape, text_rect_shape)
#
#         # Main loop
#         running = True
#         while running:
#             screen.fill(BACKGROUND_COLOR)
#             draw_graph()
#
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     running = False
#
#             keys = pygame.key.get_pressed()
#             if keys[pygame.K_LEFT] or keys[pygame.K_a]:
#                 camera_x += move_speed
#             if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
#                 camera_x -= move_speed
#             if keys[pygame.K_UP] or keys[pygame.K_w]:
#                 camera_y += move_speed
#             if keys[pygame.K_DOWN] or keys[pygame.K_s]:
#                 camera_y -= move_speed
#
#             pygame.display.flip()
#
#         pygame.quit()
#
#     def __str__(self):
#         return self.dag.__str__() + '\n' + f"Parameters: {self.param_count}" + '\n' + f"FLOPs: {self.flops}"