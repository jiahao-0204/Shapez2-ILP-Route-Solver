import pulp
import matplotlib.pyplot as plt

class DirectionalJumpRouter:
    def __init__(self, width, height, start, goals, jump_distance=4):
        self.WIDTH = width
        self.HEIGHT = height
        self.start = start
        self.goals = goals
        self.jump_distance = jump_distance
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.step_edges = []
        self._generate_step_edges()
        self.jump_edges = []
        self._generate_jump_edges()
        self.all_edges = self.step_edges + self.jump_edges
        self.model = pulp.LpProblem("JumpRouter", pulp.LpMinimize)
        self.edge_used_map = {}
        self.edge_flow_map = {}
        self.path_edges = []
        self.K = len(goals)

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
        for edge in self.all_edges:
            self.edge_used_map[edge] = pulp.LpVariable(f"bool_used_{edge}", cat='Binary')
            self.edge_flow_map[edge] = pulp.LpVariable(f"int_flow_{edge}", lowBound=0, cat='Integer')

    def add_objective(self):
        step_cost_list = [self.edge_used_map[edge] * self.step_cost for edge in self.step_edges]
        jump_cost_list = [self.edge_used_map[edge] * self.jump_cost for edge in self.jump_edges]
        self.model += pulp.lpSum(step_cost_list + jump_cost_list)

    def add_flow_constraints(self):
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model += self.edge_flow_map[edge] <= self.edge_used_map[edge] * self.K

        # Flow conservation constraints
        all_nodes = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        for node in all_nodes:
            in_flow = pulp.lpSum(self.edge_flow_map[edge] for edge in self.all_edges if edge[1] == node)
            out_flow = pulp.lpSum(self.edge_flow_map[edge] for edge in self.all_edges if edge[0] == node)

            if node == self.start:
                self.model += (out_flow - in_flow == self.K), f"start_flow_{node}"
            elif node in self.goals:
                self.model += (out_flow - in_flow == -1), f"goal_flow_{node}"
            else:
                self.model += (out_flow - in_flow == 0), f"node_flow_{node}"

    def add_directional_constraints(self):
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            # A jump from u to v in direction `direction` is only allowed
            # if there is incoming flow to u from the same direction.

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction = []
            for edge in self.all_edges:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.edge_used_map[edge])

            # create a variable that sums up the incoming edges in the same direction
            sum_of_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += self.edge_used_map[jump_edge] <= sum_of_incoming_edge_in_same_direction

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
        used_step_edges = [e for e in self.step_edges if pulp.value(self.edge_used_map[e]) == 1]
        used_jump_edges = [e for e in self.jump_edges if pulp.value(self.edge_used_map[e]) == 1]

        # Plot start and goal
        sx, sy = self.start
        for goal in self.goals:
            gx, gy = goal
            plt.scatter(gx + offset, gy + offset, c='green', marker='s', s=120, edgecolors='black')
            plt.scatter(gx + offset, gy + offset, c='black', marker='o', s=50, edgecolors='black')
        plt.scatter(sx + offset, sy + offset, c='red', marker='s', s=120, edgecolors='black', label='Start')

        for (u, v, d) in used_step_edges:
            ux, uy = u
            ax.scatter(ux + offset, uy + offset, c='black', marker='o', s=50)
            ax.plot([ux + offset, v[0] + offset], [uy + offset, v[1] + offset], c='black')

        for (u, v, d) in used_jump_edges:
            ux, uy = u
            u2x, u2y = u
            u2x += d[0] * (self.jump_distance + 1)
            u2y += d[1] * (self.jump_distance + 1)

            # if d == (0, 1):
            #     marker = '2'
            # elif d == (0, -1):
            #     marker = '1'
            # elif d == (1, 0):
            #     marker = '4'
            # elif d == (-1, 0):
            #     marker = '3'

            if d == (0, 1):
                marker = '^'
            elif d == (0, -1):
                marker = 'v'
            elif d == (1, 0):
                marker = '>'
            elif d == (-1, 0):
                marker = '<'

            ax.scatter(ux + offset, uy + offset, c='black', marker=marker, s=80)
            ax.scatter(u2x + offset, u2y + offset, c='black', marker=marker, s=80)
            ax.plot([u2x + offset, v[0] + offset], [u2y + offset, v[1] + offset], c='black')

        plt.title("ILP Path with Step & Jump Actions")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()


# Example usage
if __name__ == "__main__":
    start = (0, 0)
    goals = [(5, 13), (10, 13)]
    router = DirectionalJumpRouter(width=34, height=14, start=start, goals=goals)
    router.build_variables()
    router.add_objective()
    router.add_flow_constraints()
    router.add_directional_constraints()
    router.solve()
    router.plot()