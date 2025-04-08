import heapq
import matplotlib.pyplot as plt
import numpy as np

# Configuration
GRID_SIZE = 20
STEP_SIZE = 1
STEP_COST = 1
JUMP_COST = 3

class Action:
    def __init__(self, direction, is_jump = False, jump_size = 0, is_immediate_jump = False):
        self.direction = direction
        self.is_jump = is_jump
        self.jump_size = jump_size
        self.is_immediate_jump = is_immediate_jump

    def __str__(self):
        return f"Action (direction={self.direction}, is_jump={self.is_jump}, jump_size={self.jump_size}, is_immediate_jump={self.is_immediate_jump}), cost={self.get_cost()}"

    def get_end_location(self, start): 
        if self.is_jump:
            if self.is_immediate_jump:
                end_location = start + self.direction * (self.jump_size + 2)
            else:
                end_location = start + self.direction * (self.jump_size + 3)
        else:
            end_location = start + self.direction * STEP_SIZE
        
        # convert to tuple
        return tuple(end_location)
    
    def get_pad_location(self, start):
        if self.is_immediate_jump:
            pad_locations = [start, start + self.direction * (self.jump_size + 1)]
        else:
            pad_locations = [start + self.direction * 1, start + self.direction * (self.jump_size + 2)]
        
        # convert to tuples
        pad_locations = [tuple(tile) for tile in pad_locations]
        return pad_locations
        
    
    def get_required_tiles(self, start):
        # list of numpy arrays
        required_tiles = self.get_pad_location(start) + [self.get_end_location(start)]
        
        # convert to tuples
        required_tiles = [tuple(tile) for tile in required_tiles]
        return required_tiles

    def get_cost(self):
        if self.is_jump:
            if self.is_immediate_jump:
                return JUMP_COST - 1
            else:
                return JUMP_COST
        else:
            return STEP_COST


AVAILABLE_JUMP_SIZE = [1, 2, 3, 4]
DIRECTIONS = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
DEFAULT_ACTION_LIST = []
for direction in DIRECTIONS:
    DEFAULT_ACTION_LIST.append(Action(direction, False))
    for jump_size in AVAILABLE_JUMP_SIZE:
        DEFAULT_ACTION_LIST.append(Action(direction, True, jump_size))


def compute_jumppad_location(node, delta):
    """Return the jumper pad tiles between two nodes if it's a jump."""
    x, y = node
    dx, dy = delta
    if dx:
        direction = 1 if dx > 0 else -1
        return [(x + direction, y), (x + direction * (abs(dx) - 1), y)]
    elif dy:
        direction = 1 if dy > 0 else -1
        return [(x, y + direction), (x, y + direction * (abs(dy) - 1))]
    return []

def is_within_bounds(node):
    x, y = node
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def heuristic(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def a_star_route(start, goal, blocked_by_other_nets):
    """A* pathfinding with fixed jump support and blocked tile constraints."""
    open_heap = []
    came_from = {}
    cost_so_far = {}

    heapq.heappush(open_heap, (heuristic(start, goal), 0, start, []))
    cost_so_far[start] = 0
    came_from[start] = None

    while open_heap:
        _, current_cost, current, blocked_by_current_net = heapq.heappop(open_heap)

        if current == goal:
            break

        blocked = blocked_by_other_nets | set(blocked_by_current_net)
        blocked.discard(start)
        blocked.discard(goal)

        for action in DEFAULT_ACTION_LIST:
            next_node = action.get_end_location(current)
            required_tiles = action.get_required_tiles(current)

            if any(not is_within_bounds(loc) or loc in blocked for loc in required_tiles):
                continue

            new_cost = current_cost + action.get_cost()

            if new_cost < cost_so_far.get(next_node, float('inf')):
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                new_blocked = blocked_by_current_net + required_tiles
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(open_heap, (priority, new_cost, next_node, new_blocked))

    if goal not in came_from:
        return None, None

    # Reconstruct path and jump pads
    path = []
    pads = []
    current = goal
    while current:
        path.append(current)
        prev = came_from[current]
        if prev:
            dx, dy = current[0] - prev[0], current[1] - prev[1]
            if abs(dx) > 1 or abs(dy) > 1:
                pads.extend(compute_jumppad_location(prev, (dx, dy)))
        current = prev
    path.reverse()
    return path, pads


def draw_result(paths, pads):
    """Visualize paths and jump pads on a grid."""
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xticks(np.arange(GRID_SIZE + 1))
    plt.yticks(np.arange(GRID_SIZE + 1))
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.gca().set_aspect('equal')

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']
    for i, path in enumerate(paths):
        color = colors[i % len(colors)]
        xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in path])
        plt.plot(xs, ys, marker='o', color=color, label=f'Net {i}')
    for i, pad in enumerate(pads):
        if pad:
            color = colors[i % len(colors)]
            xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in pad])
            plt.scatter(xs, ys, color=color, s=100, marker='x', label=f'Jump Pad {i}')
    plt.legend()
    plt.title("A* Routed Paths with Fixed Jumps")
    plt.show()


if __name__ == "__main__":
    # nets = [((0, 0), (0, 5)),
    #         ((1, 0), (4, 5)),
    #         ((2, 0), (8, 5)),
    #         ((3, 0), (12, 5))]
    nets = [((0, 0), (0, 15))]

    blocked_tiles = {start for start, end in nets} | {end for start, end in nets}

    paths = []
    pads = []
    for i, (start, goal) in enumerate(nets):
        path, pad = a_star_route(start, goal, blocked_tiles)
        if path is None:
            print(f"Net {i}: no route found")
        else:
            paths.append(path)
            pads.append(pad)
            blocked_tiles.update(path)
            blocked_tiles.update(pad)
            print(f"Net {i} path (length {len(path) - 1}): {path}")

    draw_result(paths, pads)