import heapq
import matplotlib.pyplot as plt
import numpy as np
from typing import List

# Configuration
GRID_SIZE = 20
STEP_SIZE = 1
STEP_COST = 1
JUMP_COST = 3

class Action:
    def __init__(self, direction: tuple) -> None:
        self.direction = np.array(direction)

    def __str__(self) -> str:
        return (f"{self.__class__.__name__} (direction={self.direction}), "
                f"cost={self.get_cost()}")
                
    def get_end_location(self, start: tuple) -> tuple:
        raise NotImplementedError

    def get_pad_locations(self, start: tuple) -> List[tuple]:
        raise NotImplementedError

    def get_belt_locations(self, start: tuple) -> List[tuple]:
        raise NotImplementedError

    def get_blocked_tiles(self, start: tuple) -> List[tuple]:
        return self.get_pad_locations(start) + self.get_belt_locations(start)
    
    def get_required_free_tiles(self, start: tuple) -> List[tuple]:
        return self.get_pad_locations(start) + self.get_belt_locations(start) + [self.get_end_location(start)]

    def get_cost(self) -> int:
        raise NotImplementedError

class StepAction(Action):
    def get_end_location(self, start: tuple) -> tuple:
        return tuple(np.array(start) + self.direction * 1)

    def get_pad_locations(self, start: tuple) -> List[tuple]:
        return []
    
    def get_belt_locations(self, start: tuple) -> List[tuple]:
        return [start]
    
    def get_cost(self):
        return 1
    

class ImmediateJumpAction(Action):
    def __init__(self, direction, jump_size):
        super().__init__(direction)
        self.jump_size = jump_size
        
    def get_end_location(self, start: tuple) -> tuple:
        return tuple(np.array(start) + self.direction * (self.jump_size + 2))

    def get_pad_locations(self, start: tuple) -> List[tuple]:
        pads = [start, tuple(np.array(start) + self.direction * (self.jump_size + 1))]
        return pads

    def get_belt_locations(self, start: tuple) -> List[tuple]:
        return []

    def get_cost(self):
        return 2



AVAILABLE_JUMP_SIZE = [1, 2, 3, 4]
DIRECTIONS = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]

DEFAULT_ACTION_LIST = []
for direction in DIRECTIONS:
    DEFAULT_ACTION_LIST.append(StepAction(direction))
    for jump_size in AVAILABLE_JUMP_SIZE:
        DEFAULT_ACTION_LIST.append(ImmediateJumpAction(direction, jump_size))

def is_within_bounds(node):
    x, y = node
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def heuristic(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def a_star_route(start, goal, blocked_by_other_nets):
    """A* pathfinding with fixed jump support and blocked tile constraints."""

    # initialize a* variable
    open_heap = []
    came_from = {}
    action_taken_to_reach_this_node = {}
    cost_so_far = {}

    # add the start node
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start, []))
    cost_so_far[start] = 0
    came_from[start] = None

    # a* search
    while open_heap:
        _, current_cost, current, blocked_by_current_net = heapq.heappop(open_heap)

        if current == goal:
            break

        blocked = blocked_by_other_nets | set(blocked_by_current_net)
        blocked.discard(start)
        blocked.discard(goal)

        for action in DEFAULT_ACTION_LIST:

            # skip if action is jump and is not in the same direction as the previous action
            if isinstance(action, ImmediateJumpAction):
                prev_action = action_taken_to_reach_this_node.get(current)
                if prev_action and not np.array_equal(prev_action.direction, action.direction):
                    continue

            next_node = action.get_end_location(current)
            required_tiles = action.get_required_free_tiles(current)

            if any(not is_within_bounds(loc) or loc in blocked for loc in required_tiles):
                continue

            new_cost = current_cost + action.get_cost()

            if new_cost < cost_so_far.get(next_node, float('inf')):
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                action_taken_to_reach_this_node[next_node] = action
                new_blocked = blocked_by_current_net + action.get_blocked_tiles(current)
                priority = new_cost + heuristic(next_node, goal)
                heapq.heappush(open_heap, (priority, new_cost, next_node, new_blocked))

    # return none if goal is not reachable
    if goal not in came_from:
        print("Goal not reachable")
        return None, None

    # compute path, belts and pads
    path = [goal]
    belts = []
    pads = []
    cost = 0
    current = goal
    while current:
        prev = came_from[current]
        prev_action = action_taken_to_reach_this_node.get(current)
        if prev:
            path.append(prev)
            belts += prev_action.get_belt_locations(prev)
            pads += prev_action.get_pad_locations(prev)
            cost += prev_action.get_cost()
        current = prev
    path.reverse()

    # print
    print(f"Path cost: {cost}")
    
    # return
    return path, belts, pads


def draw_result(paths, belts, pads):
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
        plt.plot(xs, ys, color=color, label=f'Net {i}')
    for i, belt in enumerate(belts):
        if belt:
            color = colors[i % len(colors)]
            xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in belt])
            plt.scatter(xs, ys, color=color, s=50, marker='o', label=f'Belt {i}')
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
    nets = [((0, 0), (18, 18)),]

    blocked_tiles = {start for start, end in nets} | {end for start, end in nets}

    paths = []
    belts = []
    pads = []
    for i, (start, goal) in enumerate(nets):
        path, belt, pad = a_star_route(start, goal, blocked_tiles)
        if path is None:
            print(f"Net {i}: no route found")
        else:
            paths.append(path)
            belts.append(belt)
            pads.append(pad)
            blocked_tiles.update(belt)
            blocked_tiles.update(pad)
            print(f"Net {i} path (length {len(path) - 1}): {path}")

    draw_result(paths, belts, pads)