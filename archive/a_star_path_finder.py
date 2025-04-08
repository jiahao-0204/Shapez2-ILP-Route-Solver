import heapq
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class Action:
    def __init__(self, direction, is_jump = False, jump_size = 0, is_immediate_jump = False):
        self.step_size = 1
        self.direction = direction
        self.is_jump = is_jump
        self.jump_size = jump_size
        self.is_immediate_jump = is_immediate_jump

    def __str__(self):
        return f"Action (direction={self.direction}, is_jump={self.is_jump}, jump_size={self.jump_size}, is_immediate_jump={self.is_immediate_jump}), cost={self.get_cost()}"

    def get_end_location(self, start):
        if self.is_jump:
            if self.is_immediate_jump:
                return start + self.direction * (self.jump_size + 2)
            else:
                return start + self.direction * (self.jump_size + 3)
        else:
            return start + self.direction * self.step_size
    
    def get_pad_location(self, start):
        if self.is_immediate_jump:
            return [start, start + self.direction * (self.jump_size + 1)]
        else:
            return [start + self.direction * 1, start + self.direction * (self.jump_size + 2)]
        
    
    def get_required_tiles(self, start):
        return self.get_pad_location(start) + [self.get_end_location(start)]

    def get_cost(self):
        if self.is_jump:
            if self.is_immediate_jump:
                return 2
            else:
                return 3
        else:
            return 1


class Board:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.bottom_left = np.array([0, 0])
        self.top_right = np.array([x_size, y_size])
        self.blocked_tiles = set()

    def is_within_bounds(self, tile):
        # tile is np.array
        return (self.bottom_left[0] <= tile[0] < self.top_right[0] and
                self.bottom_left[1] <= tile[1] < self.top_right[1])

    def add_blocked_tile(self, tile):
        if 0 <= tile[0] < self.x_size and 0 <= tile[1] < self.y_size:
            self.blocked_tiles.add(tile)
    
    def add_blocked_tiles(self, tiles):
        for tile in tiles:
            self.add_blocked_tile((tile[0], tile[1]))

    def is_blocked(self, tile):
        return (tile[0], tile[1]) in self.blocked_tiles

    def is_valid(self, tile):
        return self.is_within_bounds(tile) and not self.is_blocked(tile)
    
class Net:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal
    
    def get_start(self):
        return self.start
    
    def get_goal(self):
        return self.goal
    
    def reached_goal(self, current):
        return np.array_equal(current, self.goal)

AVAILABLE_JUMP_SIZE = [1, 2, 3, 4]
DIRECTIONS = [np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
DEFAULT_ACTION_LIST = []
for direction in DIRECTIONS:
    DEFAULT_ACTION_LIST.append(Action(direction, False))
    for jump_size in AVAILABLE_JUMP_SIZE:
        DEFAULT_ACTION_LIST.append(Action(direction, True, jump_size))


MAX_ITERATIONS = 10
CONGESTION_GROW_RATE = 1
CONGESTION_DECAY_RATE = 1

def sign(x):
    return 0 if x == 0 else (1 if x > 0 else -1)

# # heuristic is different if current node is a jump vs belt
# def heuristic_belt(p, q):
#     # if they are not on the same row or column, at least one cost will incur
#     cost = 0
#     delta_x = abs(p[0] - q[0])
#     delta_y = abs(p[1] - q[1])

#     # compute cost to move in x and y direction separately
#     required_space = [x + 2 for x in AVAILABLE_JUMP_SIZE]
#     required_space.sort(reverse=True)
#     if delta_x > 0:
#         gap_x = delta_x - 1
#         cost += 1

#         for space in required_space:
#             while True:
#                 if gap_x >= space:
#                     gap_x -= space
#                     cost += 2
#                     continue
#                 else:
#                     break
#         cost += gap_x
    
#     if delta_y > 0:
#         gap_y = delta_y - 1
#         cost += 1

#         for space in required_space:
#             while True:
#                 if gap_y >= space:
#                     gap_y -= space
#                     cost += 2
#                     continue
#                 else:
#                     break
#         cost += gap_y

#     return cost

# # def heuristic_jump(p, q, direction):
# #     new_p = (p[0] + direction[0], p[1] + direction[1])
# #     return 1 + heuristic_belt(new_p, q)

def heuristic_manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def a_star_route_with_congestion_cost(net, general_board, congestion_cost_map):
    """A* pathfinding with fixed jump support and blocked tile constraints."""
    open_heap = []
    came_from = {}
    cost_so_far = {}

    start = tuple(net.get_start())
    goal = tuple(net.get_goal())

    heapq.heappush(open_heap, (heuristic_manhattan(start, goal), 0, start, general_board))
    cost_so_far[start] = 0
    came_from[start] = None

    while open_heap:
        _, current_cost, current, specific_board = heapq.heappop(open_heap)

        print(current, current_cost)

        current_vec = np.array(current)

        if np.array_equal(current_vec, goal):
            break

        for action in DEFAULT_ACTION_LIST:
            next_node = action.get_end_location(current_vec)
            next_node_tuple = tuple(next_node)

            required_tiles = [tuple(tile) for tile in action.get_required_tiles(current_vec)]

            if any(not specific_board.is_valid(np.array(tile)) for tile in required_tiles):
                continue

            new_cost = current_cost + action.get_cost()

            if new_cost < cost_so_far.get(next_node_tuple, float('inf')):
                cost_so_far[next_node_tuple] = new_cost
                came_from[next_node_tuple] = current
                specific_board.add_blocked_tiles(required_tiles)

                priority = new_cost + heuristic_manhattan(next_node_tuple, goal)
                heapq.heappush(open_heap, (priority, new_cost, next_node_tuple, specific_board))

    if goal not in came_from:
        return None, None

    # Reconstruct path and jump pads
    path = []
    pads = []
    current = goal
    while current:
        prev = came_from[current]
        path.append(current)
        current = prev

    path.reverse()
    return path, pads

def pathfinder_route(nets, global_board):
    paths = [None] * len(nets)
    pads = [None] * len(nets)
    congestion_cost_map = defaultdict(int)

    path_is_possible = True
    for iteration in range(MAX_ITERATIONS):
        # keep iterating?
        keep_going = False

        tile_usage_count = defaultdict(int)
        for i, net in enumerate(nets):
            path, pad = a_star_route_with_congestion_cost(net, global_board, congestion_cost_map)

            if path is None:
                print(f"Iteration {iteration}, Net {i}: No path is possible, ending pathfinder routing.")
                path_is_possible = False
                break
            
            if path != paths[i] or pad != pads[i]:
                keep_going = True
            paths[i] = path
            pads[i] = pad

            # add to tile usage count
            for tile in path + pad:
                tile_usage_count[tile] += 1

        # update congestion cost map 
        for tile, count in tile_usage_count.items():
            if count > 1:
                congestion_cost_map[tile] += (count - 1) * CONGESTION_GROW_RATE
            else:
                congestion_cost_map[tile] *= CONGESTION_DECAY_RATE

        if not keep_going:
            print(f"Iteration {iteration}: No further updates, ending pathfinder routing.")
            break

    if not path_is_possible:
        return None, None, None
    else:
        return paths, pads, congestion_cost_map


def draw_result(paths, pads, congestion_cost_map):
    """Visualize paths and jump pads on a grid."""
    plt.figure(figsize=(6, 6))
    plt.grid(True)
    plt.xticks(np.arange(global_board.x_size + 1))
    plt.yticks(np.arange(global_board.y_size + 1))
    plt.xlim(0, global_board.x_size)
    plt.ylim(0, global_board.y_size)
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
    
    for tile, count in congestion_cost_map.items():
        if count > 0:
            x, y = tile
            plt.text(x + 0.5, y + 0.5, str(count), fontsize=8, ha='center', va='center', color='black')

    plt.legend()
    plt.title("A* Routed Paths with Fixed Jumps")
    plt.show()


if __name__ == "__main__":
    # # nets = [((0, 0), (0, 5)),
    # #         ((1, 0), (4, 5)),
    # #         ((2, 0), (8, 5)),
    # #         ((3, 0), (12, 5))]
    # nets = [((0, 0), (3, 18))]
    nets = [
        Net((0, 0), (3, 18))
    ]

    global_board = Board(20, 20)

    paths, pads, congestion_cost_map = pathfinder_route(nets, global_board)
    if paths is None:
        print(f"No route found")

    draw_result(paths, pads, congestion_cost_map)

# print(heuristic((0, 0), (4, 10)))