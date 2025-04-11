
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
        for e in self.step_edges:
            self.x_step[e] = pulp.LpVariable(f"x_step_{e}", cat='Binary')
            self.f_step[e] = pulp.LpVariable(f"f_step_{e}", lowBound=0, cat='Integer')

        for e in self.jump_edges:
            self.x_jump[e] = pulp.LpVariable(f"x_jump_{e}", cat='Binary')
            self.f_jump[e] = pulp.LpVariable(f"f_jump_{e}", lowBound=0, cat='Integer')

    def add_objective(self):
        self.model += (
            pulp.lpSum(self.x_step[e] * self.step_cost for e in self.step_edges) +
            pulp.lpSum(self.x_jump[e] * self.jump_cost for e in self.jump_edges)
        )

    def add_flow_constraints(self):
        # Flow is avaiable if the edge is selected
        for e in self.step_edges:
            self.model += self.f_step[e] <= self.x_step[e], f"cap_step_{e}"

        for e in self.jump_edges:
            self.model += self.f_jump[e] <= self.x_jump[e], f"cap_jump_{e}"

        # Flow conservation constraints
        all_nodes = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        for node in all_nodes:
            in_flow_edges = []
            for e in self.step_edges:
                u, v, _ = e
                if v == node:
                    in_flow_edges.append(self.f_step[e])
            for e in self.jump_edges:
                u, v, _ = e
                if v == node:
                    in_flow_edges.append(self.f_jump[e])
            in_flow = pulp.lpSum(in_flow_edges)

            out_flow_edges = []
            for e in self.step_edges:
                u, v, _ = e
                if u == node:
                    out_flow_edges.append(self.f_step[e])
            for e in self.jump_edges:
                u, v, _ = e
                if u == node:
                    out_flow_edges.append(self.f_jump[e])
            out_flow = pulp.lpSum(out_flow_edges)
            
            if node == self.start:
                self.model += (out_flow - in_flow == 1), f"start_flow_{node}"
            elif node == self.goal:
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
            for edge in self.step_edges:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.x_step[edge])
            for edge in self.jump_edges:
                _, target, dir_jump = edge
                if target == u and dir_jump == direction:
                    incoming_edges_in_same_direction.append(self.x_jump[edge])

            # create a variable that sums up the incoming edges in the same direction
            existing_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += (
                self.x_jump[jump_edge] <= existing_incoming_edge_in_same_direction,
                f"directional_jump_flow_{jump_edge}"
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
        step_edges = []
        for step_edge in self.step_edges:
            if pulp.value(self.x_step[step_edge]) == 1:
                step_edges.append(step_edge)
        jump_edges = []
        for jump_edge in self.jump_edges:
            if pulp.value(self.x_jump[jump_edge]) == 1:
                jump_edges.append(jump_edge)

        # Plot start and goal
        sx, sy = self.start
        gx, gy = self.goal
        plt.scatter(sx + offset, sy + offset, c='red', marker='s', s=120, edgecolors='black', label='Start')
        plt.scatter(gx + offset, gy + offset, c='green', marker='s', s=120, edgecolors='black', label='Goal')
        plt.scatter(gx + offset, gy + offset, c='black', marker='o', s=50, edgecolors='black')

        for (u, v, d) in step_edges:
            ux, uy = u
            ax.scatter(ux + offset, uy + offset, c='black', marker='o', s=50)
            ax.plot([ux + offset, v[0] + offset], [uy + offset, v[1] + offset], c='black')

        for (u, v, d) in jump_edges:
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

# Usage
if __name__ == "__main__":
    start = (0, 0)
    goal = (20, 10)
    router = DirectionalJumpRouter(width=34, height=14, start=start, goal=goal)
    router.build_variables()
    router.add_objective()
    router.add_flow_constraints()
    router.add_directional_constraints()
    router.solve()
    router.plot()
