import heapq
import matplotlib.pyplot as plt
import numpy as np


GRID_SIZE = 15

STEP_MOVES = [(0,1), (0,-1), (1,0), (-1,0)]
STEP_COST = 1

JUMP_SIZE = 3
JUMP_MOVES = [
    (0,  (JUMP_SIZE + 3)), 
    (0, -(JUMP_SIZE + 3)),
    ( (JUMP_SIZE + 3), 0),
    (-(JUMP_SIZE + 3), 0),
    ]
JUMP_COST = 3


# Check if a jump is possible
def is_jumppad_location_clear(x, y, dx, dy, blocked):
    horizontal_jump = abs(dx) > 0
    vertical_jump = abs(dy) > 0

    if horizontal_jump: 
        direction = 1 if dx > 0 else -1
        jumppad_start_x = x + direction * 1
        jumppad_end_x = x + direction * (JUMP_SIZE + 2)
        if (jumppad_start_x, y) in blocked or (jumppad_end_x, y) in blocked:
            return False

    if vertical_jump:
        direction = 1 if dy > 0 else -1
        jumppad_start_y = y + direction * 1
        jumppad_end_y = y + direction * (JUMP_SIZE + 2)
        if (x, jumppad_start_y) in blocked or (x, jumppad_end_y) in blocked:
            return False
    return True

def compute_jumppad_location(x, y, dx, dy):
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

def is_within_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

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
    open_heap = []  # heap of (f_score, g_cost, x, y, parent)
    came_from = {}  # for path reconstruction
    g_cost = {start: 0} # cost to reach a node
    f_score = heuristic(start, goal) # cost to reach a node + heuristic to goal
    heapq.heappush(open_heap, (f_score, 0, start[0], start[1], None, []))
    
    # Perform the search
    goal_reached = False
    while open_heap:
        f, g, x, y, parent, blocked_by_current_net = heapq.heappop(open_heap)

        # skip if this node has already been visited
        if (x, y) in came_from: 
            continue

        # update came_from
        came_from[(x, y)] = parent

        # skip if we have reached the goal
        if (x, y) == goal:
            goal_reached = True
            break
        
        # Explore neighbors
        # explore 4-connected neighbors
        for delta_x, delta_y in STEP_MOVES:

            # new proposed location
            new_x, new_y = x + delta_x, y + delta_y

            # skip if the new location is already occupied
            if (new_x, new_y) in blocked_by_current_net or (new_x, new_y) in blocked_by_other_nets:
                continue

            # skip if the new location is out of bounds
            if not is_within_bounds(new_x, new_y):
                continue
            
            # compute cost to new location
            new_g = g + STEP_COST

            # update record and add to heap if this is lower than the previous cost
            if new_g < g_cost.get((new_x, new_y), float('inf')):
                g_cost[(new_x, new_y)] = new_g

                # add to heap
                new_f = new_g + heuristic((new_x, new_y), goal)
                new_blocked = blocked_by_current_net + [(new_x, new_y)]
                heapq.heappush(open_heap, (new_f, new_g, new_x, new_y, (x, y), new_blocked))

        # by jump
        for delta_x, delta_y in JUMP_MOVES:

            # new proposed location
            new_x, new_y = x + delta_x, y + delta_y

            # new proposed jumppad location
            jumppad_locations = compute_jumppad_location(x, y, delta_x, delta_y)

            # skip if the new location is already occupied
            if (new_x, new_y) in blocked_by_current_net or (new_x, new_y) in blocked_by_other_nets:
                continue

            # skip if the jumppad location is already occupied
            if any(loc in blocked_by_current_net for loc in jumppad_locations) or any(loc in blocked_by_other_nets for loc in jumppad_locations):
                continue

            # skip if the new location is out of bounds
            if not is_within_bounds(new_x, new_y):
                continue

            # skip if the jumppad location is out of bounds
            if any(not is_within_bounds(loc[0], loc[1]) for loc in jumppad_locations):
                continue

            # compute cost to new location
            new_g = g + JUMP_COST

            # update record and add to heap if this is lower than the previous cost
            if new_g < g_cost.get((new_x, new_y), float('inf')):
                g_cost[(new_x, new_y)] = new_g

                # add to heap
                new_f = new_g + heuristic((new_x, new_y), goal)
                new_blocked = blocked_by_current_net + jumppad_locations
                heapq.heappush(open_heap, (new_f, new_g, new_x, new_y, (x, y), new_blocked))
    
    # Reconstruct path if goal reached
    if not goal_reached:
        return None  # no path found
    
    path = []
    pads = []

    current_node = goal
    # Backtrack from goal to start using came_from
    while current_node is not None:
        path.append(current_node)
        previous_node = came_from[current_node]

        if previous_node is None:
            break
        

        # add jump pads if current node and previous node are not adjacent
        used_jumppads = abs(current_node[0] - previous_node[0]) > 1 or abs(current_node[1] - previous_node[1]) > 1
        if used_jumppads:
            # calculate jump pad coordinates
            x = previous_node[0]
            y = previous_node[1]
            dx = current_node[0] - previous_node[0]
            dy = current_node[1] - previous_node[1]
            jumppad_locations = compute_jumppad_location(x, y, dx, dy)
            pads.extend(jumppad_locations)
        current_node = previous_node
    path.reverse()

    return path, pads





# Example usage:
nets = [((5, 0), (5, 5)),   # Net 0: from (1,0) to (1,5)
        ((0, 3), (9, 3)),   # Net 1: from (0,2) to (5,2)
        ((4, 4), (9, 9))]   # Net 2: from (4,4) to (9,9)
blocked_tiles = set()
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
grid = np.zeros((GRID_SIZE, GRID_SIZE))
colors = ['red', 'green', 'blue']

plt.figure(figsize=(6, 6))
plt.grid(True)
plt.xlim(-1, GRID_SIZE)
plt.ylim(-1, GRID_SIZE)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.xticks(range(GRID_SIZE))
plt.yticks(range(GRID_SIZE))
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')

# Draw each path
for i, path in enumerate(paths):
    xs, ys = zip(*path)
    plt.plot(xs, ys, marker='o', color=colors[i], label=f'Net {i}')
    for x, y in path:
        plt.text(x, y, f'{i}', color='black', fontsize=8, ha='center', va='center')

# draw jump pads
for i, pad in enumerate(pads):
    if pad:  # Ensure pad is not empty
        xs, ys = zip(*pad)
        plt.scatter(xs, ys, marker='x', color=colors[i], label=f'Jump Pad {i}', s=100)
        for x, y in pad:
            plt.text(x, y, f'{i}', color='black', fontsize=8, ha='center', va='center')

plt.legend()
plt.title("Routed Paths Visualization")
plt.show()