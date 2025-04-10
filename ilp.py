import pulp

# Grid dimensions
grid_width, grid_height = 6, 6
start = (0, 0)
goal = (4, 4)

# 4-connected grid directions
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Generate edges
edges = []
for x in range(grid_width):
    for y in range(grid_height):
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                edges.append(((x, y), (nx, ny)))

# ILP model
model = pulp.LpProblem("SingleTerminalNetRouting", pulp.LpMinimize)

x_vars = { (u, v): pulp.LpVariable(f"x_{u}_{v}", cat="Binary") for (u, v) in edges }

# Objective: minimize path length
model += pulp.lpSum(x_vars[e] for e in edges)

# Flow constraints
for x in range(grid_width):
    for y in range(grid_height):
        node = (x, y)
        in_edges = [(u, v) for (u, v) in edges if v == node]
        out_edges = [(u, v) for (u, v) in edges if u == node]

        flow_in = pulp.lpSum(x_vars[(u, v)] for (u, v) in in_edges)
        flow_out = pulp.lpSum(x_vars[(u, v)] for (u, v) in out_edges)

        if node == start:
            model += (flow_out - flow_in == 1)
        elif node == goal:
            model += (flow_out - flow_in == -1)
        else:
            model += (flow_out - flow_in == 0)

# Solve
solver = pulp.PULP_CBC_CMD()
model.solve(solver)

# Output used edges
for (u, v), var in x_vars.items():
    if pulp.value(var) == 1:
        print(f"{u} -> {v}")
