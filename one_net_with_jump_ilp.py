
import pulp
import matplotlib.pyplot as plt

class DirectionalJumpRouter:
    def __init__(self, width, height, start, goal, jump_distance=4):
        self.WIDTH = width
        self.HEIGHT = height
        self.start = start
        self.goal = goal
        self.jump_distance = jump_distance
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.step_edges = []
        self._generate_step_edges()
        self.jump_edges = []
        self._generate_jump_edges()
        self.model = pulp.LpProblem("JumpRouter", pulp.LpMinimize)
        self.x_step = {}
        self.x_jump = {}
        self.f_step = {}
        self.f_jump = {}
        self.path_edges = []

        self.step_cost = 1
        self.jump_cost = 2

    def _generate_step_edges(self):
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        self.step_edges.append(((x, y), (nx, ny), (dx, dy)))

    def _generate_jump_edges(self):
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                for dx, dy in self.directions:
                    nx, ny = x + dx * (self.jump_distance + 2), y + dy * (self.jump_distance + 2)
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        self.jump_edges.append(((x, y), (nx, ny), (dx, dy)))

    def build_variables(self):
        for u, v, _ in self.step_edges:
            self.x_step[(u, v)] = pulp.LpVariable(f"x_step_{u}_{v}", cat='Binary')
            self.f_step[(u, v)] = pulp.LpVariable(f"f_step_{u}_{v}", lowBound=0, cat='Integer')

        for u, v, _ in self.jump_edges:
            self.x_jump[(u, v)] = pulp.LpVariable(f"x_jump_{u}_{v}", cat='Binary')
            self.f_jump[(u, v)] = pulp.LpVariable(f"f_jump_{u}_{v}", lowBound=0, cat='Integer')

    def add_objective(self):
        self.model += (
            pulp.lpSum(self.x_step[(u, v)] * self.step_cost for (u, v, _) in self.step_edges) +
            pulp.lpSum(self.x_jump[(u, v)] * self.jump_cost for (u, v, _) in self.jump_edges)
        )

    def add_flow_constraints(self):
        # Flow is avaiable if the edge is selected
        for (u, v, _) in self.step_edges:
            self.model += self.f_step[(u, v)] <= self.x_step[(u, v)], f"cap_step_{u}_{v}"

        for (u, v, _) in self.jump_edges:
            self.model += self.f_jump[(u, v)] <= self.x_jump[(u, v)], f"cap_jump_{u}_{v}"

        # Flow conservation constraints
        all_nodes = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        for node in all_nodes:
            in_flow = (
                pulp.lpSum(self.f_step[(u, v)] for (u, v, _) in self.step_edges if v == node) +
                pulp.lpSum(self.f_jump[(u, v)] for (u, v, _) in self.jump_edges if v == node)
            )
            out_flow = (
                pulp.lpSum(self.f_step[(u, v)] for (u, v, _) in self.step_edges if u == node) +
                pulp.lpSum(self.f_jump[(u, v)] for (u, v, _) in self.jump_edges if u == node)
            )

            if node == self.start:
                self.model += (out_flow - in_flow == 1), f"start_flow_{node}"
            elif node == self.goal:
                self.model += (out_flow - in_flow == -1), f"goal_flow_{node}"
            else:
                self.model += (out_flow - in_flow == 0), f"node_flow_{node}"

    def add_directional_constraints(self):
        for (u, v, direction) in self.jump_edges:
            # A jump from u to v in direction `direction` is only allowed
            # if there is incoming flow to u from the same direction.

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction = []
            for (prev, target, dir_step) in self.step_edges:
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.x_step[(prev, target)])
            for (prev, target, dir_jump) in self.jump_edges:
                if target == u and dir_jump == direction:
                    incoming_edges_in_same_direction.append(self.x_jump[(prev, target)])

            # create a variable that sums up the incoming edges in the same direction
            existing_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += (
                self.x_jump[(u, v)] <= existing_incoming_edge_in_same_direction,
                f"directional_jump_flow_{u}_{v}_{direction}"
            )

    def solve(self):
        solver = pulp.PULP_CBC_CMD(timeLimit=30)
        self.model.solve(solver)


    def plot(self):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.WIDTH + 1)
        ax.set_ylim(0, self.HEIGHT + 1)
        ax.set_xticks(range(self.WIDTH + 1))
        ax.set_yticks(range(self.HEIGHT + 1))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5
        step_edges = [(u, v, d) for (u, v, d) in self.step_edges if pulp.value(self.x_step[(u, v)]) == 1]
        jump_edges = [(u, v, d) for (u, v, d) in self.jump_edges if pulp.value(self.x_jump[(u, v)]) == 1]

        for (u, v, d) in step_edges:
            ux, uy = u
            ax.scatter(ux + offset, uy + offset, c='black', marker='o', s=50)

        for (u, v, d) in jump_edges:
            ux, uy = u
            u2x, u2y = u
            u2x += d[0] * (self.jump_distance + 1)
            u2y += d[1] * (self.jump_distance + 1)
            ax.scatter(ux + offset, uy + offset, c='black', marker='x', s=80)
            ax.scatter(u2x + offset, u2y + offset, c='black', marker='x', s=80)

        sx, sy = self.start
        gx, gy = self.goal
        plt.scatter(sx + offset, sy + offset, c='green', marker='s', s=120, edgecolors='black', label='Start')
        plt.scatter(gx + offset, gy + offset, c='red', marker='*', s=150, edgecolors='black', label='Goal')

        plt.title("ILP Path with Step & Jump Actions")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()

# Usage
if __name__ == "__main__":
    start = (0, 0)
    goal = (20, 0)
    router = DirectionalJumpRouter(width=34, height=14, start=start, goal=goal)
    router.build_variables()
    router.add_objective()
    router.add_flow_constraints()
    router.add_directional_constraints()
    router.solve()
    router.plot()
