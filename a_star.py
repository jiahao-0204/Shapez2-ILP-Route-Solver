import heapq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import List, Optional
from collections import defaultdict

# Configuration
BOARD_DIMENSION = (15, 6)
STEP_SIZE = 1
STEP_COST = 1
JUMP_COST = 3
AVAILABLE_JUMP_SIZE = [1, 2, 3, 4]
# AVAILABLE_JUMP_SIZE = [4]

UP = np.array([0, 1])
DOWN = np.array([0, -1])
LEFT = np.array([-1, 0])
RIGHT = np.array([1, 0])

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
    
    def is_valid_from(self, prev_action) -> bool:
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
    
    def is_valid_from(self, prev_action: Action) -> bool:
        return True

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
    
    def is_valid_from(self, prev_action: Action) -> bool:
        # If there is no previous action, this is the first move (from the start)
        if prev_action is None:
            return True

        # jump must have same direction as previous action
        return np.array_equal(prev_action.direction, self.direction)

class Keypoint:
    def __init__(self,
                 position: tuple[int, int],
                 acceptable_pad_directions: Optional[List[np.ndarray]] = None,
                 acceptable_belt_directions: Optional[List[np.ndarray]] = None) -> None:
        self.position = position
        self.acceptable_pad_directions = acceptable_pad_directions if acceptable_pad_directions is not None else [UP]
        self.acceptable_belt_directions = acceptable_belt_directions if acceptable_belt_directions is not None else [UP, DOWN, LEFT, RIGHT]
    
    def __repr__(self):
        return f"Keypoint(position={self.position})"
    
    def matches_position(self, other: tuple[int, int]) -> bool:
        return self.position == other

    def _direction_allowed(self, direction: np.ndarray, allowed: List[np.ndarray]) -> bool:
        return any(np.array_equal(direction, d) for d in allowed)

    def is_valid_action(self, action: Action) -> bool:
        if isinstance(action, StepAction):
            return self._direction_allowed(action.direction, self.acceptable_belt_directions)
        elif isinstance(action, ImmediateJumpAction):
            return self._direction_allowed(action.direction, self.acceptable_pad_directions)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

DEFAULT_ACTION_LIST = []
for direction in DIRECTIONS:
    DEFAULT_ACTION_LIST.append(StepAction(direction))
    for jump_size in AVAILABLE_JUMP_SIZE:
        DEFAULT_ACTION_LIST.append(ImmediateJumpAction(direction, jump_size))

def is_within_bounds(node):
    x, y = node
    return 0 <= x < BOARD_DIMENSION[0] and 0 <= y < BOARD_DIMENSION[1]

def heuristic(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def a_star_route(start: Keypoint, goal: Keypoint, blocked_by_other_nets: set, congestion_cost_map: Optional[dict] = None):
    """A* pathfinding with fixed jump support and blocked tile constraints."""

    # initialize a* variable
    open_heap = []
    came_from = {}
    action_taken_to_reach_this_node = defaultdict(lambda: None)
    cost_so_far = {}

    # add the start node
    heapq.heappush(open_heap, (heuristic(start.position, goal.position), 0, start.position, []))
    cost_so_far[start.position] = 0
    came_from[start.position] = None

    # a* search
    while open_heap:
        _, current_cost, current, blocked_by_current_net = heapq.heappop(open_heap)

        if goal.matches_position(current):
            break

        blocked = blocked_by_other_nets | set(blocked_by_current_net)
        blocked.discard(start.position)
        blocked.discard(goal.position)

        for action in DEFAULT_ACTION_LIST:

            # skip if action is not valid from previous action
            if not action.is_valid_from(action_taken_to_reach_this_node[current]):
                continue

            # skip if action is not valid at given location
            if start.matches_position(current) and not start.is_valid_action(action):
                continue

            # skip if action is not valid towards given location
            next_node = action.get_end_location(current)
            if goal.matches_position(next_node) and not goal.is_valid_action(action):
                continue
            
            # skip if action required tiles are blocked
            required_tiles = action.get_required_free_tiles(current)
            if any(not is_within_bounds(loc) or loc in blocked for loc in required_tiles):
                continue
            
            # compute congestion cost if congestion cost map is provided
            congestion_cost = 0
            if congestion_cost_map is not None:
                # add congestion cost for each blocked tile
                congestion_cost = sum(congestion_cost_map.get(tile, 0) for tile in action.get_blocked_tiles(current))
            
            # compute new cost
            new_cost = current_cost + action.get_cost() + congestion_cost

            # if new cost is less than the cost so far, update the cost and add to open heap
            if new_cost < cost_so_far.get(next_node, float('inf')):
                cost_so_far[next_node] = new_cost
                came_from[next_node] = current
                action_taken_to_reach_this_node[next_node] = action
                new_blocked = blocked_by_current_net + action.get_blocked_tiles(current)
                # priority = new_cost + heuristic(next_node, goal.position)
                priority = new_cost
                heapq.heappush(open_heap, (priority, new_cost, next_node, new_blocked))

    # return none if goal is not reachable
    if goal.position not in came_from:
        print("Goal not reachable")
        return None, None, None

    # compute path, belts and pads
    path = [goal.position]
    belts = []
    pads = []
    cost = 0
    current = goal.position
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

    # return
    return path, belts, pads

def compute_overlaped_tiles(belts1: Optional[List[tuple]], pads1: Optional[List[tuple]],
                           belts2: Optional[List[tuple]], pads2: Optional[List[tuple]]) -> set:
    """Compute overlaped tiles between two sets of belts and pads."""
    overlaped_tiles = set()
    if belts1:
        if belts2:
            overlaped_tiles.update(set(belts1) & set(belts2))
        if pads2:
            overlaped_tiles.update(set(belts1) & set(pads2))
    if pads1:
        if pads2:
            overlaped_tiles.update(set(pads1) & set(pads2))
        if belts2:
            overlaped_tiles.update(set(pads1) & set(belts2))
    return overlaped_tiles

def custom_routing(nets, num_of_iterations: int = 10):
    paths = [None] * len(nets)
    pads = [None] * len(nets)
    belts = [None] * len(nets)
    blocked_tiles = set()
    # add start and end position for blocked tiles
    blocked_tiles = blocked_tiles | {start.position for start, end in nets} | {end.position for start, end in nets}

    congestion_cost_map = defaultdict(float)
    for _ in range(num_of_iterations):

        for i, (start, goal) in enumerate(nets):
            blocked_tile_cost_map = defaultdict(float)
            # compute tiles taken by other nets
            tiles_taken_by_other_nets = set()
            for j in range(len(nets)):
                if j == i:
                    continue
                # compute tiles taken by other nets
                if belts[j]:
                    tiles_taken_by_other_nets.update(belts[j])
                if pads[j]:
                    tiles_taken_by_other_nets.update(pads[j])
            for tile in tiles_taken_by_other_nets:
                blocked_tile_cost_map[tile] = 1

            # combine blocked tile cost map with congestion cost map
            final_cost_map = defaultdict(float)
            for tile, cost in congestion_cost_map.items():
                final_cost_map[tile] += cost
            for tile, cost in blocked_tile_cost_map.items():
                final_cost_map[tile] += cost

            path, belt, pad = a_star_route(start, goal, blocked_tiles, final_cost_map)
            paths[i] = path
            belts[i] = belt
            pads[i] = pad

            # draw_result(nets, paths, belts, pads, final_cost_map)
        
        # compute overlaped tiles (symmetrical thus i < j)
        all_overlaped_tiles = set()
        for i in range(len(nets)):
            for j in range(i + 1, len(nets)): 
                # compute overlaped tiles
                overlaped_tiles = compute_overlaped_tiles(belts[i], pads[i], belts[j], pads[j])
                all_overlaped_tiles.update(overlaped_tiles)

        # penalize overlaped tiles
        for tile in all_overlaped_tiles:
            congestion_cost_map[tile] += 1

        # decay non overlaped tiles
        for tile, cost in congestion_cost_map.items():
            if tile not in all_overlaped_tiles:
                congestion_cost_map[tile] = cost * 0.9

        # draw_result(nets, paths, belts, pads)
    return paths, belts, pads

def sequential_routing(nets):
    paths = []
    belts = []
    pads = []
    blocked_tiles = set()
    # add start and end position for blocked tiles
    blocked_tiles = blocked_tiles | {start.position for start, end in nets} | {end.position for start, end in nets}

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

    return paths, belts, pads

def pathfinder_routing(nets, num_of_iterations: int):
    paths = [None] * len(nets)
    pads = [None] * len(nets)
    belts = [None] * len(nets)
    blocked_tiles = set()
    # add start and end position for blocked tiles
    blocked_tiles = blocked_tiles | {start.position for start, end in nets} | {end.position for start, end in nets}

    congestion_cost_map = defaultdict(float)
    for _ in range(num_of_iterations):

        for i, (start, goal) in enumerate(nets):
            path, belt, pad = a_star_route(start, goal, blocked_tiles, congestion_cost_map)
            paths[i] = path
            belts[i] = belt
            pads[i] = pad
        
        # update tile usage count
        tile_usage_count = defaultdict(int)
        for belt in belts:
            if belt:
                for tile in belt:
                    tile_usage_count[tile] += 1
        for pad in pads:
            if pad:
                for tile in pad:
                    tile_usage_count[tile] += 1
        
        # update congestion cost map
        for tile, count in tile_usage_count.items():
            if count > 1:
                congestion_cost_map[tile] += count - 1

        draw_result(nets, paths, belts, pads, congestion_cost_map)
    return paths, belts, pads

def draw_result(nets, paths, belts, pads, congestion_cost_map: Optional[dict] = None):
    """Visualize paths and jump pads on a grid."""
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xticks(np.arange(BOARD_DIMENSION[0] + 1))
    plt.yticks(np.arange(BOARD_DIMENSION[1] + 1))
    plt.xlim(0, BOARD_DIMENSION[0])
    plt.ylim(0, BOARD_DIMENSION[1])
    plt.gca().set_aspect('equal')

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']

    # draw paths
    for i, path in enumerate(paths):
        # skip if path is None
        if path is None or len(path) == 0:
            continue
        
        color = colors[i % len(colors)]
        xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in path])
        plt.plot(xs, ys, color=color, label=f'Net {i}', zorder=1)

    # draw belts
    for i, belt in enumerate(belts):
        # skip if belt is None
        if belt is None or len(belt) == 0:
            continue
        
        color = colors[i % len(colors)]
        xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in belt])
        plt.scatter(xs, ys, color=color, s=50, marker='o', label=f'Belt {i}')
        
    # draw pads
    for i, pad in enumerate(pads):
        # skip if pad is None
        if pad is None or len(pad) == 0:
            continue
        
        color = colors[i % len(colors)]
        xs, ys = zip(*[(x + 0.5, y + 0.5) for x, y in pad])
        plt.scatter(xs, ys, color=color, s=100, marker='x', label=f'Jump Pad {i}')
    
    # draw keypoint
    for i, (start, goal) in enumerate(nets):
        color = colors[i % len(colors)]

        # # draw start as square
        # plt.scatter(start.position[0] + 0.5, start.position[1] + 0.5,
        #     color=color, s=20, marker='s', edgecolors='black', linewidths=0.8, label=f'Start {i}', zorder=2)

        # draw goal as star
        plt.scatter(goal.position[0] + 0.5, goal.position[1] + 0.5,
            color=color, s=100, marker='*', edgecolors='black', linewidths=0.8, label=f'Goal {i}', zorder=2)

    # draw congestion cost map
    if congestion_cost_map:
        for tile, cost in congestion_cost_map.items():
            if cost > 0:
                x, y = tile
                plt.text(x + 0.5, y + 0.5, str(cost), ha='center', va='center', fontsize=8, color='black')
    # legend    
    custom_legend = [
        Line2D([0], [0], marker='*', color='grey', markersize=12, linestyle='None', label='Goal'),
        Line2D([0], [0], marker='x', color='grey', markersize=8, linestyle='None', label='Jump Pad'),
        Line2D([0], [0], marker='o', color='grey', markerfacecolor='grey', markersize=6, linestyle='None', label='Belt'),
        Line2D([0], [0], color='grey', label='Path'),
    ]
    plt.legend(handles=custom_legend)

    plt.title("Shapez2: Sequential Dijkstra Routing -- Jiahao")
    plt.show()


if __name__ == "__main__":
    # nets = [((0, 0), (0, 5)),
    #         ((1, 0), (4, 5)),
    #         ((2, 0), (8, 5)),
    #         ((3, 0), (12, 5))]

    acceptable_belt_directions = [UP, LEFT, RIGHT]
    # acceptable_belt_directions = [UP]

    nets = [
        (Keypoint((2, 0)), Keypoint((8, 5), acceptable_belt_directions=acceptable_belt_directions)),
        (Keypoint((3, 0)), Keypoint((12, 5), acceptable_belt_directions=acceptable_belt_directions)),
        (Keypoint((1, 0)), Keypoint((4, 5), acceptable_belt_directions=acceptable_belt_directions)),
        (Keypoint((0, 0)), Keypoint((0, 5), acceptable_belt_directions=acceptable_belt_directions)),
    ]

    # paths, belts, pads = pathfinder_routing(nets, num_of_iterations=10)
    # paths, belts, pads = sequential_routing(nets)
    paths, belts, pads = custom_routing(nets)

    draw_result(nets, paths, belts, pads)