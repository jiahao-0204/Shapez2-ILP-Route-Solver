
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
        self.edges = self._generate_step_edges()
        self.jump_edges = self._generate_jump_edges()
        self.model = pulp.LpProblem("JumpRouter", pulp.LpMinimize)
        self.x_step = {}
        self.x_jump = {}
        self.f_step = {}
        self.f_jump = {}
        self.path_edges = []

    def _generate_step_edges(self):
        edges = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edges.append(((x, y), (nx, ny), (dx, dy)))
        return edges

    def _generate_jump_edges(self):
        edges = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                for dx, dy in self.directions:
                    nx, ny = x + dx * self.jump_distance, y + dy * self.jump_distance
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edges.append(((x, y), (nx, ny), (dx, dy)))
        return edges

    def build_variables(self):
        for u, v, d in self.edges:
            self.x_step[(u, v)] = pulp.LpVariable(f"x_step_{u}_{v}", cat='Binary')
            self.f_step[(u, v)] = pulp.LpVariable(f"f_step_{u}_{v}", lowBound=0, cat='Integer')

        for u, v, d in self.jump_edges:
            self.x_jump[(u, v, d)] = pulp.LpVariable(f"x_jump_{u}_{v}_{d}", cat='Binary')
            self.f_jump[(u, v, d)] = pulp.LpVariable(f"f_jump_{u}_{v}_{d}", lowBound=0, cat='Integer')

    def add_objective(self):
        self.model += (
            pulp.lpSum(self.x_step[(u, v)] for (u, v, _) in self.edges) +
            2 * pulp.lpSum(self.x_jump[(u, v, d)] for (u, v, d) in self.jump_edges)
        )

    def add_flow_constraints(self):
        all_nodes = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        for node in all_nodes:
            in_flow = (
                pulp.lpSum(self.f_step[(u, v)] for (u, v, _) in self.edges if v == node) +
                pulp.lpSum(self.f_jump[(u, v, d)] for (u, v, d) in self.jump_edges if v == node)
            )
            out_flow = (
                pulp.lpSum(self.f_step[(u, v)] for (u, v, _) in self.edges if u == node) +
                pulp.lpSum(self.f_jump[(u, v, d)] for (u, v, d) in self.jump_edges if u == node)
            )

            if node == self.start:
                self.model += (out_flow - in_flow == 1), f"start_flow_{node}"
            elif node == self.goal:
                self.model += (out_flow - in_flow == -1), f"goal_flow_{node}"
            else:
                self.model += (out_flow - in_flow == 0), f"node_flow_{node}"

    def add_capacity_and_direction_constraints(self):
        for (u, v, d) in self.edges:
            self.model += self.f_step[(u, v)] <= self.x_step[(u, v)], f"cap_step_{u}_{v}"

        for (u, v, d) in self.jump_edges:
            self.model += self.f_jump[(u, v, d)] <= self.x_jump[(u, v, d)], f"cap_jump_{u}_{v}"

            # Directional flow constraint: jump at u in direction d only if incoming flow is in same direction
            dx, dy = d
            allowed_in = [
                (u2, u, d) for (u2, _, d2) in self.jump_edges if _ == u and d2 == d
            ] + [
                (u2, u) for (u2, _, d2) in self.edges if _ == u and d2 == d
            ]
            incoming = (
                pulp.lpSum(self.f_step[e] for e in allowed_in if e in self.f_step) +
                pulp.lpSum(self.f_jump[e] for e in allowed_in if e in self.f_jump)
            )
            self.model += self.f_jump[(u, v, d)] <= incoming, f"dir_jump_{u}_{v}_{d}"

    def solve(self):
        solver = pulp.PULP_CBC_CMD(timeLimit=30)
        self.model.solve(solver)


    def plot(self):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(-0.5, self.WIDTH - 0.5)
        ax.set_ylim(-0.5, self.HEIGHT - 0.5)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5
        step_edges = [(u, v, d) for (u, v, d) in self.edges if pulp.value(self.x_step[(u, v)]) == 1]
        jump_edges = [(u, v, d) for (u, v, d) in self.jump_edges if pulp.value(self.x_jump[(u, v, d)]) == 1]

        for (u, v, d) in step_edges:
            ux, uy = u
            ax.scatter(ux + offset, uy + offset, c='black', marker='o', s=50)

        for (u, v, d) in jump_edges:
            ux, uy = u
            u2x, u2y = u
            u2x += d[0] * (self.jump_distance - 1)
            u2y += d[1] * (self.jump_distance - 1)
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
    goal = (10, 10)
    router = DirectionalJumpRouter(width=15, height=15, start=start, goal=goal)
    router.build_variables()
    router.add_objective()
    router.add_flow_constraints()
    router.add_capacity_and_direction_constraints()
    router.solve()
    router.plot()
