import torch
from collections import deque
from line_profiler import profile
from tqdm import tqdm
import heapq
import math
import pygame

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
        '''Evaluate the model on the test set. Returns loss, accuracy, results tuple'''
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

    def get_height_width(self):
        '''Conducts a single forward pass of the graph and outputs a tuple (height, width)'''
        height, width = 0, 0

        # Min heap (by layer) for the children of the root
        heap = []
        count = 0  # Tie-breaker counter. Otherwise we get an error trying to compare nodes
        for child in self.dag.graph[self.dag.root]:
            # Push a tuple of (layer, counter, node) into the heap
            heapq.heappush(heap, (child.layer, count, child))
            count += 1

        # Pop nodes in order of increasing layer
        curr_layer_count = 0 # Number of nodes on this layer
        prev_layer = 0
        while heap:
            _, _, node = heapq.heappop(heap)

            if node.layer == prev_layer:
                curr_layer_count += 1
            else:
                curr_layer_count = 1
                prev_layer = node.layer
                height = node.layer

            width = max(width, curr_layer_count)

            for child in self.dag.graph[node]:
                heapq.heappush(heap, (child.layer, count, child))
                count += 1

        return height, width

    def visualize(self):
        '''Visualize the network with camera movement using arrow keys or WASD.'''
        pygame.init()

        # Constants
        NODE_RADIUS = 40
        BACKGROUND_COLOR = (255, 255, 255)
        NODE_COLOR = (189, 140, 245)
        EDGE_COLOR = (0, 0, 0)
        TEXT_COLOR = (0, 0, 0)
        ARROW_SIZE = 10
        WIDTH = 1200
        HEIGHT = 800

        # Display
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Network Visualization")

        # Movement variables
        camera_x = 0
        camera_y = 0
        move_speed = 10

        node_positions = {self.dag.root: (50, HEIGHT // 2)}
        heap = []
        count = 0
        for child in self.dag.graph[self.dag.root]:
            heapq.heappush(heap, (child.layer, count, child))
            count += 1

        curr_layer = []
        prev_layer = 0
        while heap:
            _, _, node = heapq.heappop(heap)
            if node.layer == prev_layer:
                curr_layer.append(node)
            else:
                for i, x in enumerate(curr_layer):
                    node_positions[x] = (50 + prev_layer * 100, 50 + i * 50)
                curr_layer = [node]
                prev_layer = node.layer
            curr_layer.append(node)
            for child in self.dag.graph[node]:
                heapq.heappush(heap, (child.layer, count, child))
                count += 1
        for i, x in enumerate(curr_layer):
            node_positions[x] = (50 + prev_layer * 100, 50 + i * 50)

        def draw_arrow(start, end, color, arrow_size=ARROW_SIZE):
            pygame.draw.line(screen, color, start, end, 2)
            angle = math.atan2(end[1] - start[1], end[0] - start[0])
            x1 = end[0] - arrow_size * math.cos(angle - math.pi / 6)
            y1 = end[1] - arrow_size * math.sin(angle - math.pi / 6)
            x2 = end[0] - arrow_size * math.cos(angle + math.pi / 6)
            y2 = end[1] - arrow_size * math.sin(angle + math.pi / 6)
            pygame.draw.polygon(screen, color, [(end[0], end[1]), (x1, y1), (x2, y2)])

        def draw_graph():
            screen.fill(BACKGROUND_COLOR)
            for node, neighbors in self.dag.graph.items():
                start_pos = (node_positions[node][0] + camera_x, node_positions[node][1] + camera_y)
                for neighbor in neighbors:
                    if neighbor in node_positions:
                        end_pos = (node_positions[neighbor][0] + camera_x, node_positions[neighbor][1] + camera_y)
                        draw_arrow(start_pos, end_pos, EDGE_COLOR)
            for node, pos in node_positions.items():
                adjusted_pos = (pos[0] + camera_x, pos[1] + camera_y)
                pygame.draw.circle(screen, NODE_COLOR, adjusted_pos, NODE_RADIUS)
                font = pygame.font.Font(None, 20)
                text_surface_desc = font.render(f"{node.desc}", True, TEXT_COLOR)
                text_surface_shape = font.render(f"{node.shape}", True, TEXT_COLOR)
                text_rect_desc = text_surface_desc.get_rect(center=(adjusted_pos[0], adjusted_pos[1] - 10))
                text_rect_shape = text_surface_shape.get_rect(center=(adjusted_pos[0], adjusted_pos[1] + 10))
                screen.blit(text_surface_desc, text_rect_desc)
                screen.blit(text_surface_shape, text_rect_shape)

        # Main loop
        running = True
        while running:
            screen.fill(BACKGROUND_COLOR)
            draw_graph()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                camera_x += move_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                camera_x -= move_speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                camera_y += move_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                camera_y -= move_speed

            pygame.display.flip()

        pygame.quit()

    def __str__(self):
        return self.dag.__str__() + '\n' + f"Parameters: {self.param_count}" + '\n' + f"FLOPs: {self.flops}"

