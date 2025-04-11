import pulp
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
JUMP_COST = 2
STEP_COST = 1

Node = Tuple[int, int]
Edge = Tuple[Node, Node, Tuple[int, int]]  # (start_node, end_node, direction)

class DirectionalJumpRouter:
    def __init__(self, width, height, start, goals, jump_distance=4):

        # Input parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.start = start
        self.goals = goals
        self.jump_distance = jump_distance

        # Internal variables
        self.K = len(goals)
        self.all_nodes: List[Node] = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        self.all_edges: List[Edge] = []
        self.step_edges: List[Edge] = []
        self.jump_edges: List[Edge] = []
        self.node_related_step_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_jump_edges: Dict[Node, List[Edge]] = defaultdict(list)
        for node in self.all_nodes:
            x, y = node
            for dx, dy in DIRECTIONS:

                # Step edge
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.step_edges.append(edge)
                    self.node_related_step_edges[node].append(edge)
                
                # Jump edge
                nx, ny = x + dx * (self.jump_distance + 2), y + dy * (self.jump_distance + 2)
                jx, jy = x + dx * (self.jump_distance + 1), y + dy * (self.jump_distance + 1)
                pad_node = (jx, jy)
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.jump_edges.append(edge)
                    self.node_related_jump_edges[node].append(edge)
                    self.node_related_jump_edges[pad_node].append(edge)



        # Optimization
        self.model = pulp.LpProblem("JumpRouter", pulp.LpMinimize)
        
        # Dynamic variables
        self.is_edge_used: Dict[Edge, pulp.LpVariable] = {edge: pulp.LpVariable(f"bool_used_{edge}", cat='Binary') for edge in self.all_edges}
        self.is_node_used_by_step_edge: Dict[Node, pulp.LpVariable] = {node: pulp.LpVariable(f"step_node_used_{node}", cat='Binary') for node in self.all_nodes}
        self.is_node_used_by_jump_edge: Dict[Node, pulp.LpVariable] = {node: pulp.LpVariable(f"jump_node_used_{node}", cat='Binary') for node in self.all_nodes}
        self.edge_flow_value: Dict[Edge, pulp.LpVariable] = {edge: pulp.LpVariable(f"int_flow_{edge}", lowBound=0, cat='Integer') for edge in self.all_edges}

        # Dynamic computes
        self.dynamic_compute_is_node_used_by()
        
        # Objective function
        self.add_objective()

        # Constraints
        self.add_constraints()

        # Solve
        self.solve()

        # Plot
        self.plot()

    def dynamic_compute_is_node_used_by(self):
        for node in self.all_nodes:
            if self.node_related_step_edges[node]:
                self.model += pulp.lpSum(self.is_edge_used[e] for e in self.node_related_step_edges[node]) <= len(self.node_related_step_edges[node]) *  self.is_node_used_by_step_edge[node]
            if self.node_related_jump_edges[node]:
                self.model += pulp.lpSum(self.is_edge_used[e] for e in self.node_related_jump_edges[node]) <= len(self.node_related_jump_edges[node]) * self.is_node_used_by_jump_edge[node]

    def add_objective(self):
        step_cost_list = [self.is_edge_used[edge] * STEP_COST for edge in self.step_edges]
        jump_cost_list = [self.is_edge_used[edge] * JUMP_COST for edge in self.jump_edges]
        self.model += pulp.lpSum(step_cost_list + jump_cost_list)

    def add_constraints(self):
        self.add_flow_constraints()
        self.add_overlap_constraints()
        self.add_directional_constraints()
        self.add_goal_action_constraints()

    def add_flow_constraints(self):
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model += self.edge_flow_value[edge] <= self.is_edge_used[edge] * self.K

        # Flow conservation constraints
        for node in self.all_nodes:
            in_flow = pulp.lpSum(self.edge_flow_value[edge] for edge in self.all_edges if edge[1] == node)
            out_flow = pulp.lpSum(self.edge_flow_value[edge] for edge in self.all_edges if edge[0] == node)

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
                    incoming_edges_in_same_direction.append(self.is_edge_used[edge])

            # create a variable that sums up the incoming edges in the same direction
            sum_of_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += self.is_edge_used[jump_edge] <= sum_of_incoming_edge_in_same_direction

    def add_overlap_constraints(self):
        for node in self.all_nodes:
            # Constraint: a node cannot be used by both step and jump
            self.model += (
                self.is_node_used_by_step_edge[node] + self.is_node_used_by_jump_edge[node] <= 1,
                f"no_overlap_at_node_{node}"
            )

    def add_goal_action_constraints(self):
        # no action is to be taken at the goal nodes
        for goal in self.goals:
            self.model += pulp.lpSum(self.is_node_used_by_jump_edge[goal] + self.is_node_used_by_step_edge[goal]) == 0, f"no_action_at_goal_{goal}"

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
        used_step_edges = [e for e in self.step_edges if pulp.value(self.is_edge_used[e]) == 1]
        used_jump_edges = [e for e in self.jump_edges if pulp.value(self.is_edge_used[e]) == 1]

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