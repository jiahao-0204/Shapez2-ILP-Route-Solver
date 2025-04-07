import heapq
import matplotlib.pyplot as plt
import numpy as np

def compute_jumppad_location(node, delta):
    x, y = node
    dx, dy = delta

    horizontal_jump = abs(dx) > 0
    vertical_jump = abs(dy) > 0

    if horizontal_jump: 
        direction = 1 if dx > 0 else -1
        jumppad_start_x = x + direction * 1
        jumppad_end_x = x + direction * (dx - 1)
        return [(jumppad_start_x, y), (jumppad_end_x, y)]

    if vertical_jump:
        direction = 1 if dy > 0 else -1
        jumppad_start_y = y + direction * 1
        jumppad_end_y = y + direction * (dy - 1)
        return [(x, jumppad_start_y), (x, jumppad_end_y)]
    
    return []

def is_within_bounds(node):
    return 0 <= node[0] < GRID_SIZE and 0 <= node[1] < GRID_SIZE

def is_end_location_clear(x, y, blocked):
    # Check if the end location is clear
    if (x, y) in blocked:
        return False
    return True

# Manhattan distance heuristic
def heuristic(p, q):
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def a_star_route(start, goal, blocked_by_other_nets):
    """Find shortest path from start to goal on a grid of given size using A*.
    `blocked` is a set of coordinates that cannot be used (occupied by other nets)."""
    
    # A* search structures
    heap = [] # priority queue for open nodes (which node to explore next using f_score)
    came_from = {}  # a previous to current node mapping for backtracking
    cost_so_far = {} # cost to reach a node

    # start node
    priority = heuristic(start, goal)
    cost = 0
    previous_node_location = None
    blocked_by_current_net = []

    # add start node to heap
    heapq.heappush(heap, (priority, cost, start, blocked_by_current_net))
    came_from[start] = previous_node_location
    cost_so_far[start] = cost

    # Perform the search
    goal_reached = False
    while heap:

        # get the node with the lowest f_score / highest priority from the heap and explore its neighbors
        _, current_cost, current_node_location, blocked_by_current_net = heapq.heappop(heap)

        # skip if we have reached the goal
        if current_node_location == goal:
            goal_reached = True
            break
        
        # compute blocked locations
        blocked = blocked_by_other_nets.copy()
        blocked.update(blocked_by_current_net)
        # remove the start and goal from the blocked list
        blocked.discard(start)
        blocked.discard(goal)

        # explore possible next node locations
        for delta in MOVES:

            # next node location
            next_node_location = (current_node_location[0] + delta[0], current_node_location[1] + delta[1])

            # check if require a jumppad
            require_jumppad = abs(delta[0]) > 1 or abs(delta[1]) > 1

            # compute all locations required for this move
            all_locations = [next_node_location]
            if require_jumppad:
                jumppad_locations = compute_jumppad_location(current_node_location, delta)
                all_locations += jumppad_locations

            # skip if any of the locations are already occupied
            if any(loc in blocked for loc in all_locations):
                continue

            # skip if any of the locations are out of bounds
            if any(not is_within_bounds(loc) for loc in all_locations):
                continue

            # compute cost to new node location
            move_cost = STEP_COST if not require_jumppad else JUMP_COST
            next_cost = current_cost + move_cost

            # skip if there are existing lower cost paths to the next node
            if next_cost > cost_so_far.get(next_node_location, float('inf')):
                continue
            
            # update cost_so_far at the next node location
            cost_so_far[next_node_location] = next_cost

            # update came_from to reflect the new cost / path taken
            came_from[next_node_location] = current_node_location

            # add next node to heap to explore
            priority = next_cost + heuristic(next_node_location, goal)
            new_blocked = blocked_by_current_net + all_locations
            heapq.heappush(heap, (priority, next_cost, next_node_location, new_blocked))
            
    
    # skip if we have not reached the goal
    if not goal_reached:
        return None, None  # no path found
    
    # Backtrack from goal to start using came_from to reconstruct the path and pads
    path = []
    pads = []
    current_node = goal
    while current_node is not None:
        path.append(current_node)
        previous_node_location = came_from[current_node]

        # skip if we have reached the start
        if previous_node_location is None:
            break

        # add jump pads if current node and previous node are not adjacent
        used_jumppads = abs(current_node[0] - previous_node_location[0]) > 1 or abs(current_node[1] - previous_node_location[1]) > 1
        if used_jumppads:
            # calculate jump pad coordinates
            delta = (current_node[0] - previous_node_location[0], current_node[1] - previous_node_location[1])
            jumppad_locations = compute_jumppad_location(previous_node_location, delta)
            pads.extend(jumppad_locations)
        current_node = previous_node_location
    path.reverse()

    # return
    return path, pads

def draw_result(paths, pads):
    """Draw the grid with paths and jump pads centered in grid cells."""

    plt.figure(figsize=(6, 6))
    plt.grid(True, which='both')

    # Draw borders
    plt.plot([0, GRID_SIZE], [GRID_SIZE, GRID_SIZE], color='black', lw=2)
    plt.plot([GRID_SIZE, GRID_SIZE], [0, GRID_SIZE], color='black', lw=2)
    plt.plot([0, 0], [0, GRID_SIZE], color='black', lw=2)
    plt.plot([0, GRID_SIZE], [0, 0], color='black', lw=2)

    # Set ticks to be at the **center** of grid cells
    tick_positions = np.arange(GRID_SIZE+1)
    plt.xticks(tick_positions, labels=[str(i) for i in range(GRID_SIZE+1)])
    plt.yticks(tick_positions, labels=[str(i) for i in range(GRID_SIZE+1)])
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.gca().set_aspect('equal')

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']

    for i, path in enumerate(paths):
        xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in path])  # Shift to center
        color = colors[i % len(colors)]
        plt.plot(xs, ys, marker='o', color=color, label=f'Net {i}')

    for i, pad in enumerate(pads):
        if pad:
            xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in pad])
            color = colors[i % len(colors)]
            plt.scatter(xs, ys, marker='x', s=100, color=color, label=f'Jump Pad {i}')

    plt.legend()
    plt.title("Routed Paths Centered in Grid Cells")
    plt.show()


if __name__ == "__main__":
    GRID_SIZE = 20
    JUMP_SIZE = 4

    # moves and costs, theese shouldn't be changed
    STEP_COST = 1
    JUMP_COST = 3
    STEP_MOVES = [(0,1), (0,-1), (1,0), (-1,0)]    
    JUMP_MOVES = [(0,  (JUMP_SIZE + 3)), (0, -(JUMP_SIZE + 3)), ( (JUMP_SIZE + 3), 0), (-(JUMP_SIZE + 3), 0)]
    MOVES = STEP_MOVES + JUMP_MOVES

    # problem setup
    nets = [((0, 0), (0, 5)),
            ((1, 0), (4, 5)),
            ((2, 0), (8, 5)),
            ((3, 0), (12, 5))]
    
    # put the start and goal into the blocked tiles
    blocked_tiles = set()
    for start, goal in nets:
        blocked_tiles.add(start)
        blocked_tiles.add(goal)
    blocked_tiles.add((4, 6))

    # program execution
    paths = []
    pads = []
    for i, (start, goal) in enumerate(nets):
        path, pad = a_star_route(start, goal, blocked_tiles)
        if path is None:
            print(f"Net {i}: no route found")
        else:
            paths.append(path)
            pads.append(pad)
            # mark this path's tiles as occupied
            blocked_tiles.update(path)  
            # mark jump pads as occupied
            blocked_tiles.update(pad)
            print(f"Net {i} path (length {len(path)-1}): {path}")


    # Create the grid and mark the paths
    draw_result(paths, pads)