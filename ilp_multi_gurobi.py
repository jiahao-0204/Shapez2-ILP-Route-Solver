import pulp
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List
from matplotlib.lines import Line2D

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
JUMP_COST = 2
STEP_COST = 1

Node = Tuple[int, int] # (x, y)
Edge = Tuple[Node, Node, Tuple[int, int]]  # (start_node, end_node, direction)

class DirectionalJumpRouter:
    def __init__(self, width, height, nets, jump_distances: List[int] = [4], timelimit: int = 60):

        # allow multiple start

        # Input parameters
        self.WIDTH = width
        self.HEIGHT = height

        self.num_nets = len(nets)
        self.start: Dict[int, Tuple[int, int]] = {}
        self.goals: Dict[int, List[Tuple[int, int]]] = {}
        for i, (start, goals) in enumerate(nets):
            self.start[i] = start
            self.goals[i] = goals

        self.jump_distances = jump_distances
        self.timelimit = timelimit




        # Internal variables
        self.K: Dict[int, int] = {}
        for i, (start, goals) in enumerate(nets):
            self.K[i] = len(goals)

        # all nodes
        self.all_nodes: List[Node] = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                self.all_nodes.append((x, y))
        
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
                
                for jump_distance in self.jump_distances:
                    nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
                    jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
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
        self.is_edge_used: Dict[int, Dict[Edge, pulp.LpVariable]] = {}
        for i in range(self.num_nets):
            self.is_edge_used[i] = {edge: pulp.LpVariable(f"edge_used_{i}_{edge}", cat='Binary') for edge in self.all_edges}    
        
        self.is_node_used_by_net: Dict[int, Dict[Node, pulp.LpVariable]] = self.dynamic_compute_is_node_used_by()

        # Objective function
        self.add_objective()

        # Constraints
        self.add_constraints()

        # Solve
        self.solve()

        # Plot
        self.plot()

    def dynamic_compute_is_node_used_by(self):
        is_node_used_by_net: Dict[int, Dict[Node, pulp.LpVariable]] = defaultdict(lambda: defaultdict(pulp.LpVariable))
        for i in range(self.num_nets):
            for node in self.all_nodes:

                step_edges_from_node = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
                jump_edges_related_to_node = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]
                
                is_node_used_by_net[i][node] = pulp.LpVariable(f"node_used_by_net_{i}_{node}", cat='Binary')    
                # self.model += len(step_edges_from_node) * is_node_used_by_net[i][node] >= pulp.lpSum(step_edges_from_node) + len(step_edges_from_node) * pulp.lpSum(jump_edges_related_to_node)
                self.model += is_node_used_by_net[i][node] >= pulp.lpSum(step_edges_from_node) / len(step_edges_from_node) + pulp.lpSum(jump_edges_related_to_node)
        return is_node_used_by_net

    def add_objective(self):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        self.model += pulp.lpSum(step_cost_list + jump_cost_list)

    def add_constraints(self):
        for i in range(self.num_nets):
            self.add_flow_constraints_v2(i)
            self.add_directional_constraints(i)
            self.add_overlap_and_one_jump_constraints(i)

        self.add_symmetry_constraints()
        self.add_goal_action_constraints()
        self.add_net_overlap_constraints()

    def add_flow_constraints(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, pulp.LpVariable]] = {}
        self.edge_flow_value[i] = {edge: pulp.LpVariable(f"edge_flow_value_{i}_{edge}", cat='Integer', lowBound=0, upBound=self.K[i]) for edge in self.all_edges}
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model += self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * self.K[i]
            self.model += self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge]

        # Flow conservation constraints
        for node in self.all_nodes:
            in_flow = pulp.lpSum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = pulp.lpSum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)

            if node == self.start[i]:
                self.model += (out_flow - in_flow == self.K[i])
            elif node in self.goals[i]:
                self.model += (out_flow - in_flow == -1)
            else:
                self.model += (out_flow - in_flow == 0)

    def add_flow_constraints_v2(self, i):
        for node in self.all_nodes:
            in_flow = [self.is_edge_used[i][edge] for edge in self.all_edges if edge[1] == node]
            out_flow = [self.is_edge_used[i][edge] for edge in self.all_edges if edge[0] == node]

            if node == self.start[i]:
                self.model += sum(in_flow) == 0
                self.model += sum(out_flow) == 1
            elif node in self.goals[i]:
                self.model += sum(in_flow) == 1
                self.model += sum(out_flow) == 0
            else:
                # if have in_flow, then must have out_flow
                self.model += sum(in_flow) / len(in_flow) <= sum(out_flow)
                # self.model += sum(in_flow) <= sum(out_flow) * len(in_flow)
                # if have out_flow, then must have in_flow
                self.model += sum(out_flow) / len(out_flow) <= sum(in_flow)
                # self.model += sum(out_flow) <= sum(in_flow) * len(out_flow)

        # the flow in and flow out must not be cyclic
        max_level = self.WIDTH + self.HEIGHT  # rough upper bound for longest path
        node_level = {}

        for node in self.all_nodes:
            node_level[node] = pulp.LpVariable(f"node_level_{i}_{node}", lowBound=0, upBound=max_level, cat='Integer')
                
        # Acyclic constraint using topological levels
        M = max_level + 1
        for edge in self.all_edges:
            u, v, _ = edge
            self.model += node_level[v] >= node_level[u] + 1 - M * (1 - self.is_edge_used[i][edge])

    def add_directional_constraints(self, i):
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            # A jump from u to v in direction `direction` is only allowed
            # if there is incoming flow to u from the same direction.

            # if u is at start, then only up jump is allowed
            if u == self.start[i]:
                if direction == (0, 1):
                    continue
                else:
                    self.model += self.is_edge_used[i][jump_edge] == 0
                    continue

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction = []
            for edge in self.all_edges:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.is_edge_used[i][edge])

            # create a variable that sums up the incoming edges in the same direction
            sum_of_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model += self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction

    def add_overlap_and_one_jump_constraints(self, i):
        for node in self.all_nodes:
            # list of all step edges from this node
            step_edges_from_node = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]

            # list of all jump edges of this node
            jump_edges_related_to_node = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]

            # the constraint
            self.model += (
                # pulp.lpSum(step_edges_from_node) + len(step_edges_from_node) * pulp.lpSum(jump_edges_related_to_node) <= len(step_edges_from_node)
                pulp.lpSum(step_edges_from_node) / len(step_edges_from_node) + pulp.lpSum(jump_edges_related_to_node) <= 1
            )

    # def add_overlap_constraints_v3(self):
    #     for node in self.all_nodes[0]:
    #         step_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)
    #         jump_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)

    #         for i in range(self.num_nets):
    #             step_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
    #             jump_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]

    #         # the constraint
    #         self.model += (
    #             pulp.lpSum(pulp.lpSum(step_edges_of_net[i]) / len(step_edges_of_net[i]) for i in range(self.num_nets)) + 
    #             pulp.lpSum(pulp.lpSum(jump_edges_of_net[i]) for i in range(self.num_nets)) <= 1)

    # def add_overlap_constraints(self, i):
    #     for node in self.all_nodes:
    #         # Constraint: a node cannot be used by both step and jump
    #         self.model += (
    #             self.is_node_used_by_step_edge[i][node] + self.is_node_used_by_jump_edge[i][node] <= 1
    #         )

    # def add_one_jump_constraints(self, i):
    #     # at most one jump edge allowed in each node
    #     for node in self.all_nodes:
    #         # create a variable that sums up the jump edges in the same node
    #         num_of_jump_edges_per_node = pulp.lpSum(self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node])

    #         # Enforce that at most one jump edge is allowed in each node
    #         self.model += num_of_jump_edges_per_node <= 1


    def add_goal_action_constraints(self):
        # no action is to be taken at the goal nodes for any net
        for i in range(self.num_nets):
            for j in range(self.num_nets):
                for goal in self.goals[j]:
                    self.model += self.is_node_used_by_net[i][goal] == 0

    def add_net_overlap_constraints(self):
        # no overlap between nets
        for node in self.all_nodes:
            list_of_nets_using_node = []
            for i in range(self.num_nets):
                list_of_nets_using_node.append(self.is_node_used_by_net[i][node])
            
            # constraint: at most one net can use a node
            self.model += pulp.lpSum(list_of_nets_using_node) <= 1

    def add_jump_pad_implication(self):
        # if a jump edge is used, then the corresponding jump pad must be used
        for i in range(self.num_nets):
            for jump_edge in self.jump_edges:
                u, v, direction = jump_edge
                # A jump from u to v in direction `direction` is only allowed
                # if there is incoming flow to u from the same direction.

                # Collect all edges (step and jump) that go into `u` from direction `direction`
                incoming_edges_in_same_direction = []
                for edge in self.all_edges:
                    _, target, dir_step = edge
                    if target == u and dir_step == direction:
                        incoming_edges_in_same_direction.append(self.is_edge_used[i][edge])

                # create a variable that sums up the incoming edges in the same direction
                sum_of_incoming_edge_in_same_direction = pulp.lpSum(incoming_edges_in_same_direction)

                # Enforce that jump pad flow is only allowed if incoming flow matches direction
                self.model += self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction

    def add_symmetry_constraints(self):
        # net i should reflex net K-i
        for i in range(int(self.num_nets / 2)):
            j = self.num_nets - i - 1
            for edge in self.all_edges:
                u, v, d = edge
                sym_ux = self.WIDTH - u[0]
                sym_suy = u[1]
                sym_svx = self.WIDTH - v[0]
                sym_svy = v[1]
                sym_u = (sym_ux, sym_suy)
                sym_v = (sym_svx, sym_svy)
                sym_d = (-d[0], d[1])
                if ((sym_u, sym_v, sym_d) in self.all_edges[j]):
                    self.model += self.is_edge_used[i][edge] == self.is_edge_used[j][(sym_u, sym_v, sym_d)]

    def solve(self):
        # solver = pulp.PULP_CBC_CMD(timeLimit=self.timelimit)
        # solver = pulp.GUROBI_CMD(timeLimit=self.timelimit, options=[("MIPFocus", 1)])
        solver = pulp.GUROBI_CMD(timeLimit=self.timelimit)
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

            used_step_edges = [e for e in self.step_edges if pulp.value(self.is_edge_used[i][e]) == 1]
            used_jump_edges = [e for e in self.jump_edges if pulp.value(self.is_edge_used[i][e]) == 1]

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
                jump_distance = max(abs(v[0] - u[0]), abs(v[1] - u[1])) - 2
                u2x += d[0] * (jump_distance + 1)
                u2y += d[1] * (jump_distance + 1)

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
        

        plt.title("Shapez2: Routing using Integer Linear Programming (ILP) -- Jiahao")
        custom_legend = [
            Line2D([0], [0], marker='s', color='grey', markersize=9, markeredgecolor='black', linestyle='None', label='Start/Goal'),
            Line2D([0], [0], marker='^', color='grey', markersize=8, markeredgecolor='black', linestyle='None', label='Jump Pad'),
            Line2D([0], [0], marker='o', color='grey', markersize=7, markeredgecolor='black', linestyle='None', label='Belt'),
        ]
        plt.legend(handles=custom_legend)

        plt.show()


# Example usage
if __name__ == "__main__":
    nets = [
        ((5, 0), [(1, 6), (3, 6), (5, 6), (7, 6)]),
        # ((6, 0), [(9, 6), (11, 6), (13, 6), (15, 6)]),
        # ((7, 0), [(17, 6), (19, 6), (21, 6), (23, 6)]),
        # ((8, 0), [(25, 6), (27, 6), (29, 6), (31, 6)]),
        # ((25, 0), [(2, 6), (4, 6), (6, 6), (8, 6)]),
        # ((26, 0), [(10, 6), (12, 6), (14, 6), (16, 6)]),
        # ((27, 0), [(18, 6), (20, 6), (22, 6), (24, 6)]),
        ((28, 0), [(26, 6), (28, 6), (30, 6), (32, 6)]),
        ]
    router = DirectionalJumpRouter(width=33, height=7, nets=nets, jump_distances= [1, 2, 3, 4], timelimit = 360)