import heapq
import matplotlib.pyplot as plt
import numpy as np

# Configuration
GRID_SIZE = 20
AVAILABLE_JUMP_SIZE = [1, 2, 3, 4]
STEP_COST = 1
JUMP_COST = 3
MOVES = [(0, 1), (0, -1), (1, 0), (-1, 0)]
for jump_size in AVAILABLE_JUMP_SIZE:
    MOVES += [(0, jump_size + 3), (0, -(jump_size + 3)),
              (jump_size + 3, 0), (-(jump_size + 3), 0)]


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

        for delta in MOVES:
            next_node = (current[0] + delta[0], current[1] + delta[1])
            is_jump = abs(delta[0]) > 1 or abs(delta[1]) > 1
            required_tiles = [next_node] + compute_jumppad_location(current, delta) if is_jump else [next_node]

            if any(not is_within_bounds(loc) or loc in blocked for loc in required_tiles):
                continue

            move_cost = JUMP_COST if is_jump else STEP_COST
            new_cost = current_cost + move_cost

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