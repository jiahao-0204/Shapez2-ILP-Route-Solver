import heapq
import random
import matplotlib.pyplot as plt
import numpy as np

# Define grid size
GRID_SIZE = 10

# Define step and jump moves
STEP_MOVES = [(0,1), (0,-1), (1,0), (-1,0)]
JUMP_MOVES = [(0,7), (0,-7), (7,0), (-7,0)]
JUMP_COST = 1
STEP_COST = 1

# Manhattan heuristic
def heuristic(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

# Check if a jump is possible
def is_jump_clear(x, y, dx, dy, blocked):
    # horizontal jump
    if dx != 0: 
        direction = 1 if dx > 0 else -1
        jumppad_start_location = x + direction * 1
        jumppad_end_location = x + direction * 6
        if (jumppad_start_location, y) in blocked or (jumppad_end_location, y) in blocked:
            return False
        # if jumppad_start_location < 0 or jumppad_end_location >= GRID_SIZE:
        #     return False
    # vertical jump
    else:
        direction = 1 if dy > 0 else -1
        jumppad_start_location = y + direction * 1
        jumppad_end_location = y + direction * 6
        if (x, jumppad_start_location) in blocked or (x, jumppad_end_location) in blocked:
            return False
        # if jumppad_start_location < 0 or jumppad_end_location >= GRID_SIZE:
        #     return False
    return True

# Pathfinder routing with negotiated congestion
def pathfinder_route(start, goal, base_costs, congestion, grid_size):

    # use a heap to store visited nodes
    open_heap = []

    # came_from is a dict storing from: and to: coordinates
    came_from = {}


    g_cost = {start: 0}
    f_score = heuristic(start, goal)
    heapq.heappush(open_heap, (f_score, 0, start[0], start[1], None))

    while open_heap:
        # get the node with the lowest f_score from the heap and start the search
        f, g, x, y, parent = heapq.heappop(open_heap)

        # Check if we have already visited this node
        if (x, y) in came_from:
            continue


        came_from[(x, y)] = parent
        if (x, y) == goal:
            break

        # compute cost to step and add to heap
        for delta_x, delta_y in STEP_MOVES:
            new_x, new_y = x + delta_x, y + delta_y
            # check if resultant location is within bounds
            resultant_location_within_bounds = 0 <= new_x < grid_size and 0 <= new_y < grid_size

            if resultant_location_within_bounds:
                tile_cost = base_costs.get((new_x, new_y), 1) + congestion.get((new_x, new_y), 0)
                new_g = g + STEP_COST + tile_cost
                if new_g < g_cost.get((new_x, new_y), float('inf')):
                    g_cost[(new_x, new_y)] = new_g
                    new_f = new_g + heuristic((new_x, new_y), goal)
                    heapq.heappush(open_heap, (new_f, new_g, new_x, new_y, (x, y)))

        # compute cost to jump and add to heap
        for delta_x, delta_y in JUMP_MOVES:
            new_x, new_y = x + delta_x, y + delta_y

            # check if resultant location is within bounds
            resultant_location_within_bounds = 0 <= new_x < grid_size and 0 <= new_y < grid_size

            if 0 <= new_x < grid_size and 0 <= new_y < grid_size: #  and is_jump_clear(x, y, delta_x, delta_y, base_costs)
                jump_path = [(x+i*delta_x//5, y+i*delta_y//5) for i in range(6)]
                jump_cost = sum(base_costs.get(p, 1) + congestion.get(p, 0) for p in jump_path)
                new_g = g + JUMP_COST + jump_cost
                if new_g < g_cost.get((new_x, new_y), float('inf')):
                    g_cost[(new_x, new_y)] = new_g
                    new_f = new_g + heuristic((new_x, new_y), goal)
                    heapq.heappush(open_heap, (new_f, new_g, new_x, new_y, (x, y)))

    # Reconstruct path
    node = goal
    if node not in came_from:
        return None
    path = []
    while node:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path

# Apply Pathfinder routing for multiple nets with congestion negotiation
def route_nets_pathfinder(nets, grid_size, iterations=5):
    base_costs = {}
    final_paths = []

    # repeat for a number of iterations
    for _ in range(iterations):
        congestion = {}
        intermediate_paths = []

        # for each net, find a path and update congestion
        for start, goal in nets:

            # generate path
            path = pathfinder_route(start, goal, base_costs, congestion, grid_size)
            if path is None:
                return []  # Failed to route
            intermediate_paths.append(path)

            # add cost to congested path
            for p in path:
                congestion[p] = congestion.get(p, 0) + 1
        
        # update base costs based on congestion
        base_costs = {p: 1 + congestion[p] for p in congestion}

        # update the final paths
        final_paths = intermediate_paths

    # return the final paths
    return final_paths

# Example nets
nets = [((1, 0), (1, 5)), ((0, 2), (5, 2)), ((4, 4), (9, 9))]
paths = route_nets_pathfinder(nets, GRID_SIZE)

# Visualization
def visualize_paths(paths, grid_size):
    grid = np.zeros((grid_size, grid_size))
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal')

    for i, path in enumerate(paths):
        xs, ys = zip(*path)
        plt.plot(xs, ys, marker='o', color=colors[i % len(colors)], label=f'Net {i}')
        for x, y in path:
            plt.text(x, y, f'{i}', color='black', fontsize=8, ha='center', va='center')

    plt.legend()
    plt.title("PathFinder Routed Paths")
    plt.show()

visualize_paths(paths, GRID_SIZE)
