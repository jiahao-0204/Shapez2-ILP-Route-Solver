
import pulp
import matplotlib.pyplot as plt

class MultiTerminalRouter:
    def __init__(self, width, height, source, goals):
        self.WIDTH = width
        self.HEIGHT = height
        self.source = source
        self.goals = goals
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.step_cost = 1
        self.edges = self._generate_edges()
        self.model = pulp.LpProblem("OneNetMultipleGoals", pulp.LpMinimize)
        self.step_used = {}
        self.f_vars = {}
        self.path_edges = []

    def _generate_edges(self):
        edges = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edges.append(((x, y), (nx, ny)))
        return edges

    def build_variables(self):
        for u, v in self.edges:
            self.step_used[(u, v)] = pulp.LpVariable(f"x_{u}_{v}", cat='Binary')
            self.f_vars[(u, v)] = pulp.LpVariable(f"f_{u}_{v}", lowBound=0, cat='Integer')

    def add_objective(self):
        self.model += pulp.lpSum(self.step_used[(u, v)] * self.step_cost for (u, v) in self.edges)

    def add_flow_constraints(self):
        k = len(self.goals)
        all_nodes = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        for node in all_nodes:
            in_edges = [(u, v) for (u, v) in self.edges if v == node]
            out_edges = [(u, v) for (u, v) in self.edges if u == node]
            flow_in = pulp.lpSum(self.f_vars[(u, v)] for (u, v) in in_edges)
            flow_out = pulp.lpSum(self.f_vars[(u, v)] for (u, v) in out_edges)

            if node == self.source:
                self.model += (flow_out - flow_in == k), f"source_flow_{node}"
            elif node in self.goals:
                self.model += (flow_out - flow_in == -1), f"goal_flow_{node}"
            else:
                self.model += (flow_out - flow_in == 0), f"intermediate_flow_{node}"

    def add_capacity_constraints(self):
        k = len(self.goals)
        for (u, v) in self.edges:
            self.model += self.f_vars[(u, v)] <= k * self.step_used[(u, v)], f"capacity_{u}_{v}"

    def solve(self):
        solver = pulp.PULP_CBC_CMD(timeLimit=3)
        self.model.solve(solver)
        self.path_edges = [(u, v) for (u, v), var in self.step_used.items() if pulp.value(var) == 1]
        return self.path_edges

    def plot_paths(self):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(-0.5, self.WIDTH - 0.5)
        ax.set_ylim(-0.5, self.HEIGHT - 0.5)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5
        for (u, v) in self.path_edges:
            ux, uy = u
            vx, vy = v
            ax.plot([ux + offset, vx + offset], [uy + offset, vy + offset], 'b-', linewidth=2)

        sx, sy = self.source
        plt.scatter(sx + offset, sy + offset, c='green', marker='s', s=120, edgecolors='black', label='Start')

        for i, (gx, gy) in enumerate(self.goals):
            plt.scatter(gx + offset, gy + offset, c='red', marker='*', s=150, edgecolors='black', label=f'Goal {i}' if i == 0 else "")

        plt.title("ILP Routing: One Start to Multiple Goals")
        plt.legend()
        plt.show()


# Usage
if __name__ == "__main__":
    source = (5, 0)
    goals = [(0, 5), (5, 5), (10, 10)]
    router = MultiTerminalRouter(width=34, height=14, source=source, goals=goals)
    router.build_variables()
    router.add_objective()
    router.add_flow_constraints()
    router.add_capacity_constraints()
    router.solve()
    router.plot_paths()
