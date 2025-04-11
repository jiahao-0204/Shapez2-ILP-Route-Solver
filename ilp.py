import pulp
import matplotlib.pyplot as plt

# Grid size
WIDTH, HEIGHT = 34, 14

# Define nets: each as (start, goal)
nets = [((0, 0), (2, 2)), ((1, 0), (1, 4))]

# Define directions for movement
UP = (0, 1)
DOWN = (0, -1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]


# ILP model
model = pulp.LpProblem("MultiNetRouting", pulp.LpMinimize)

# action has costs
# i need to minimize the total cost from actions 

# Generate valid action edges (from tile u to tile v)
edges = []
for x in range(WIDTH):
    for y in range(HEIGHT):
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                edges.append(((x, y), (nx, ny)))

# Decision variables: x_{i, u, v} = 1 if net i uses edge (u,v)
x_vars = {}
for i in range(len(nets)):
    for u, v in edges:
        x_vars[(i, u, v)] = pulp.LpVariable(f"x_{i}_{u}_{v}", cat='Binary')

# Objective: minimize total path length
model += pulp.lpSum(x_vars[(i, u, v)] for i in range(len(nets)) for u, v in edges)

# Flow constraints for each net
for i, (start, goal) in enumerate(nets):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            node = (x, y)
            in_edges = [(u, v) for (u, v) in edges if v == node]
            out_edges = [(u, v) for (u, v) in edges if u == node]

            flow_in = pulp.lpSum(x_vars[(i, u, v)] for u, v in in_edges)
            flow_out = pulp.lpSum(x_vars[(i, u, v)] for u, v in out_edges)

            if node == start:
                model += (flow_out - flow_in == 1), f"net_{i}_start_{node}"
            elif node == goal:
                model += (flow_out - flow_in == -1), f"net_{i}_goal_{node}"
            else:
                model += (flow_out - flow_in == 0), f"net_{i}_node_{node}"

# each node can only be used by one net
node_vars = {}  # node_vars[(i, n)] = 1 if net i uses node n
for i in range(len(nets)):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            node = (x, y)
            node_vars[(i, node)] = pulp.LpVariable(f"node_{i}_{node}", cat='Binary')

for i in range(len(nets)):
    for x in range(WIDTH):
        for y in range(HEIGHT):
            node = (x, y)
            in_edges = [(u, v) for (u, v) in edges if v == node]
            out_edges = [(u, v) for (u, v) in edges if u == node]
            related_edges = in_edges + out_edges
            model += (
                pulp.lpSum(x_vars[(i, u, v)] for (u, v) in related_edges) <= 2 * node_vars[(i, node)],
                f"node_link_{i}_{node}"
            )

for x in range(WIDTH):
    for y in range(HEIGHT):
        node = (x, y)
        model += (
            pulp.lpSum(node_vars[(i, node)] for i in range(len(nets))) <= 1,
            f"node_exclusivity_{node}"
        )


# Solve the ILP
solver = pulp.PULP_CBC_CMD()
model.solve(solver)

# Output the paths
paths = [[] for _ in nets]
for (i, u, v), var in x_vars.items():
    if pulp.value(var) == 1:
        paths[i].append((u, v))

def plot_paths(paths, nets):
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_xlim(0, WIDTH + 1)
    ax.set_ylim(0, HEIGHT + 1)
    ax.set_xticks(range(WIDTH + 1))
    ax.set_yticks(range(HEIGHT + 1))
    ax.set_aspect('equal')
    ax.grid(True)

    location_offset = 0.5  # Center within grid cells

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    for i, path in enumerate(paths):
        color = colors[i % len(colors)]
        for (u, v) in path:
            ux, uy = u
            vx, vy = v
            ax.plot(
                [ux + location_offset, vx + location_offset],
                [uy + location_offset, vy + location_offset],
                color=color,
                linewidth=2,
                label=f'Net {i}' if (u, v) == path[0] else ""
            )

        # Draw start and goal
        sx, sy = nets[i][0]
        gx, gy = nets[i][1]
        plt.scatter(sx + location_offset, sy + location_offset, c=color, marker='s', s=100, edgecolors='black', label=f'Start {i}')
        plt.scatter(gx + location_offset, gy + location_offset, c=color, marker='*', s=150, edgecolors='black', label=f'Goal {i}')

    plt.legend()
    plt.title("ILP Multi-Net Path Planning")
    plt.show()


# Call visualizer
plot_paths(paths, nets)