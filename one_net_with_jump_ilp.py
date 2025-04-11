import pulp
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
JUMP_COST = 2
STEP_COST = 1

Node = Tuple[int, int] # (x, y)
Edge = Tuple[Node, Node, Tuple[int, int]]  # (start_node, end_node, direction)

class DirectionalJumpRouter:
    def __init__(self, width, height, nets, jump_distance: int = 4):

        # Input parameters
        self.WIDTH = width
        self.HEIGHT = height

        self.num_nets = len(nets)
        self.start: Dict[int, Tuple[int, int]] = {}
        self.goals: Dict[int, List[Tuple[int, int]]] = {}
        for i, (start, goals) in enumerate(nets):
            self.start[i] = start
            self.goals[i] = goals

        self.jump_distance = jump_distance




        # Internal variables
        self.K: Dict[int, int] = {}
        for i, (start, goals) in enumerate(nets):
            self.K[i] = len(goals)

        self.all_nodes: Dict[int, List[Node]] = {}
        for i in range(self.num_nets):
            self.all_nodes[i] = []
            for x in range(self.WIDTH):
                for y in range(self.HEIGHT):
                    self.all_nodes[i].append((x, y))
        
        self.all_edges: Dict[int, List[Edge]] = {}
        self.step_edges: Dict[int, List[Edge]] = {}
        self.jump_edges: Dict[int, List[Edge]] = {}
        self.node_related_step_edges: Dict[int, Dict[Node, List[Edge]]] = defaultdict(lambda: defaultdict(list))
        self.node_related_jump_edges: Dict[int, Dict[Node, List[Edge]]] = defaultdict(lambda: defaultdict(list))
        for i in range(self.num_nets):
            self.all_edges[i] = []
            self.step_edges[i] = []
            self.jump_edges[i] = []
            self.node_related_step_edges[i] = defaultdict(list)
            self.node_related_jump_edges[i] = defaultdict(list)
            for node in self.all_nodes[i]:
                x, y = node
                for dx, dy in DIRECTIONS:

                    # Step edge
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges[i].append(edge)
                        self.step_edges[i].append(edge)
                        self.node_related_step_edges[i][node].append(edge)
                    
                    # Jump edge
                    nx, ny = x + dx * (self.jump_distance + 2), y + dy * (self.jump_distance + 2)
                    jx, jy = x + dx * (self.jump_distance + 1), y + dy * (self.jump_distance + 1)
                    pad_node = (jx, jy)
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges[i].append(edge)
                        self.jump_edges[i].append(edge)
                        self.node_related_jump_edges[i][node].append(edge)
                        self.node_related_jump_edges[i][pad_node].append(edge)

        # Optimization
        self.model = pulp.LpProblem("JumpRouter", pulp.LpMinimize)
        
        # Dynamic variables
        self.is_edge_used: Dict[int, Dict[Edge, pulp.LpVariable]] = {}
        self.edge_flow_value: Dict[int, Dict[Edge, pulp.LpVariable]] = {}
        self.is_node_used_by_step_edge: Dict[int, Dict[Node, pulp.LpVariable]] = {}
        self.is_node_used_by_jump_edge: Dict[int, Dict[Node, pulp.LpVariable]] = {}
        for i in range(self.num_nets):
            self.is_edge_used[i] = {edge: pulp.LpVariable(f"edge_used_{i}_{edge}", cat='Binary') for edge in self.all_edges[i]}
            self.edge_flow_value[i] = {edge: pulp.LpVariable(f"edge_flow_value_{i}_{edge}", lowBound=0) for edge in self.all_edges[i]}
            self.is_node_used_by_step_edge[i] = {node: pulp.LpVariable(f"node_used_by_step_edge_{i}_{node}", cat='Binary') for node in self.all_nodes[i]}
            self.is_node_used_by_jump_edge[i] = {node: pulp.LpVariable(f"node_used_by_jump_edge_{i}_{node}", cat='Binary') for node in self.all_nodes[i]}
        
        
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
        for i in range(self.num_nets):
            for node in self.all_nodes[i]:
                if self.node_related_step_edges[i][node]:
                    self.model += pulp.lpSum(self.is_edge_used[i][e] for e in self.node_related_step_edges[i][node]) <= len(self.node_related_step_edges[i][node]) *  self.is_node_used_by_step_edge[i][node]
                if self.node_related_jump_edges[i][node]:
                    self.model += pulp.lpSum(self.is_edge_used[i][e] for e in self.node_related_jump_edges[i][node]) <= len(self.node_related_jump_edges[i][node]) * self.is_node_used_by_jump_edge[i][node]

    def add_objective(self):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges[i]]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges[i]]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        self.model += pulp.lpSum(step_cost_list + jump_cost_list)

    def add_constraints(self):
        for i in range(self.num_nets):
            self.add_flow_constraints(i)
            self.add_directional_constraints(i)
            self.add_overlap_constraints(i)
            self.add_goal_action_constraints(i)

    def add_flow_constraints(self, i):
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges[i]:
            self.model += self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * self.K[i]
            self.model += self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge]

        # Flow conservation constraints
        for node in self.all_nodes[i]:
            in_flow = pulp.lpSum(self.edge_flow_value[i][edge] for edge in self.all_edges[i] if edge[1] == node)
            out_flow = pulp.lpSum(self.edge_flow_value[i][edge] for edge in self.all_edges[i] if edge[0] == node)

            if node == self.start[i]:
                self.model += (out_flow - in_flow == self.K[i])
            elif node in self.goals[i]:
                self.model += (out_flow - in_flow == -1)
            else:
                self.model += (out_flow - in_flow == 0)

    def add_directional_constraints(self, i):
        for jump_edge in self.jump_edges[i]:
            u, v, direction = jump_edge
            # A jump from u to v in direction `direction` is only allowed
            # if there is incoming flow to u from the same direction.

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction = []
            for edge in self.all_edges[i]:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.is_edge_used[i][edge])

            # create a variable that sums up the incoming edges in the same direction
            sum_of_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction

    def add_overlap_constraints(self, i):
        for node in self.all_nodes[i]:
            # Constraint: a node cannot be used by both step and jump
            self.model += (
                self.is_node_used_by_step_edge[i][node] + self.is_node_used_by_jump_edge[i][node] <= 1
            )

    def add_goal_action_constraints(self, i):
        # no action is to be taken at the goal nodes
        for goal in self.goals[i]:
            self.model += pulp.lpSum(self.is_node_used_by_jump_edge[i][goal] + self.is_node_used_by_step_edge[i][goal]) == 0

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

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']    
        for i in range(self.num_nets):
            color = colors[i % len(colors)]

            used_step_edges = [e for e in self.step_edges[i] if pulp.value(self.is_edge_used[i][e]) == 1]
            used_jump_edges = [e for e in self.jump_edges[i] if pulp.value(self.is_edge_used[i][e]) == 1]

            # Plot start and goal
            sx, sy = self.start[i]
            for goal in self.goals[i]:
                gx, gy = goal
                plt.scatter(gx + offset, gy + offset, c=color, marker='s', s=120, edgecolors='black', zorder = 0)
                plt.scatter(gx + offset, gy + offset, c=color, marker='o', s=50, edgecolors='black', zorder = 2)
            plt.scatter(sx + offset, sy + offset, c=color, marker='s', s=120, edgecolors='black', label='Start', zorder = 0)

            # plot step circule and line
            for (u, v, d) in used_step_edges:
                ux, uy = u
                ax.plot([ux + offset, v[0] + offset], [uy + offset, v[1] + offset], c='black', zorder = 1)
                ax.scatter(ux + offset, uy + offset, c=color, marker='o', s=50, edgecolors='black', zorder = 2)

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

                ax.plot([u2x + offset, v[0] + offset], [u2y + offset, v[1] + offset], c='black', zorder = 1)
                ax.scatter(ux + offset, uy + offset, c=color, marker=marker, s=80, edgecolors='black', zorder = 2)
                ax.scatter(u2x + offset, u2y + offset, c=color, marker=marker, s=80, edgecolors='black', zorder = 2)
        

        plt.title("ILP Path with Step & Jump Actions")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.show()


# Example usage
if __name__ == "__main__":
    nets = [
        ((0, 0), [(5, 13), (10, 13)]),
        ((10, 0), [(15, 13), (18, 13)]),
        ]
    router = DirectionalJumpRouter(width=34, height=14, nets=nets, jump_distance=4)