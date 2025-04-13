from ortools.sat.python import cp_model
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

        self.starts: List[Tuple[int, int]] = []
        self.goals: List[Tuple[int, int]] = []
        for (start, goal) in nets:
            self.starts.append(start)
            self.goals.append(goal)

        self.jump_distances = jump_distances
        self.timelimit = timelimit
        self.num_nets = len(self.starts)

        # Optimization
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

        # new set up, each node have edges, each edge have a bool var, each node have a list of bool var to indicate which net is used at this node


        # all nodes
        self.all_nodes: List[Node] = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                self.all_nodes.append((x, y))
        
        # edges at each node
        self.all_edges: List[Edge] = []
        self.step_edges: List[Edge] = []
        self.jump_edges: List[Edge] = []
        self.is_edge_used: Dict[Edge, cp_model.BoolVarT] = {}
        self.step_edges_at_node: Dict[Node, List[Edge]] = defaultdict(list)
        self.jump_edges_at_node: Dict[Node, List[Edge]] = defaultdict(list)
        for node in self.all_nodes:
            x, y = node
            for dx, dy in DIRECTIONS:
                # step edge
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.step_edges.append(edge)
                    self.is_edge_used[edge] = self.model.NewBoolVar(f"edge_used_{node}_{edge}")
                    self.step_edges_at_node[node].append(edge)
                
                # jump edges
                for jump_distance in self.jump_distances:
                    nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
                    jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
                    pad_node = (jx, jy)
                    if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges.append(edge)
                        self.jump_edges.append(edge)
                        self.is_edge_used[edge] = self.model.NewBoolVar(f"edge_used_{node}_{edge}")
                        self.jump_edges_at_node[node].append(edge)
                        self.jump_edges_at_node[pad_node].append(edge)

        # nets at each node
        self.nets_at_each_node: Dict[Node, List[cp_model.BoolVarT]] = defaultdict(list)
        for node in self.all_nodes:
            for i in range(self.num_nets):
                self.nets_at_each_node[node].append(self.model.NewBoolVar(f"node_used_by_net_{i}_{node}"))
        
        # step edge used at node
        self.is_step_edge_used_at_node: Dict[Node, cp_model.BoolVarT] = defaultdict(cp_model.BoolVarT)
        for node in self.all_nodes:
            # list of all step edges from this node
            edges_bool_list = [self.is_edge_used[edge] for edge in self.step_edges_at_node[node]]

            # add a variable that is true if any step edge is used for this node
            self.is_step_edge_used_at_node[node] = self.model.NewBoolVar(f"step_edge_used_{node}")
            self.model.AddBoolOr(edges_bool_list).OnlyEnforceIf(self.is_step_edge_used_at_node[node])
            self.model.AddBoolAnd([edge.Not() for edge in edges_bool_list]).OnlyEnforceIf(self.is_step_edge_used_at_node[node].Not())

        # node used at node
        self.is_node_used_at_node: Dict[Node, cp_model.BoolVarT] = defaultdict(cp_model.BoolVarT)
        for node in self.all_nodes:
            # list of all edges from this node
            edges_bool_list = [self.is_edge_used[edge] for edge in self.step_edges_at_node[node] + self.jump_edges_at_node[node]]

            # add a variable that is true if any edge is used for this node
            self.is_node_used_at_node[node] = self.model.NewBoolVar(f"node_used_{node}")
            self.model.AddBoolOr(edges_bool_list).OnlyEnforceIf(self.is_node_used_at_node[node])
            self.model.AddBoolAnd([edge.Not() for edge in edges_bool_list]).OnlyEnforceIf(self.is_node_used_at_node[node].Not())

            if node in self.goals:
                self.model.Add(self.is_node_used_at_node[node] == 0)
        
        # self.is_node_used_by_net: Dict[int, Dict[Node, cp_model.BoolVarT]] = self.dynamic_compute_is_node_used_by_net()

        # Objective function
        self.add_objective_by_node_used()

        # Constraints
        self.add_constraints()

        # Solve
        self.solve()

        # Plot
        self.plot()

    def dynamic_compute_is_node_used_by_net(self):
        is_node_used_by_net: Dict[int, Dict[Node, cp_model.BoolVarT]] = defaultdict(lambda: defaultdict(cp_model.BoolVarT))
        for i in range(self.num_nets):
            for node in self.all_nodes[i]:

                is_step_edge_used = self.is_step_edge_used_at_node[i][node]
                jump_edges_related_to_node = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[i][node]]
                
                is_node_used_by_net[i][node] = self.model.NewBoolVar(f"node_used_by_net_{i}_{node}")
                self.model.AddBoolOr([is_step_edge_used] + jump_edges_related_to_node).OnlyEnforceIf(is_node_used_by_net[i][node])
                self.model.AddBoolAnd([is_step_edge_used.Not()] + [edge.Not() for edge in jump_edges_related_to_node]).OnlyEnforceIf(is_node_used_by_net[i][node].Not())
        return is_node_used_by_net

    def add_objective(self):
        tile_used = []
        for i in range(self.num_nets):
            for node in self.all_nodes[i]:
                tile_used.append(self.is_node_used_by_net[i][node])
        self.model.Minimize(sum(tile_used))

    def add_objective_by_node_used(self):
        # minimize the number of node used
        node_bool_list = []
        for node in self.all_nodes:        
            node_bool_list.append(self.is_node_used_at_node[node])

        self.model.Minimize(sum(node_bool_list))

    def add_constraints(self):
        self.add_simple_flow_constraints()
        self.add_directional_constraints()

        self.add_overlap_and_one_jump_constraints()

        # self.add_symmetry_constraints()
        self.add_goal_action_constraints()
        # self.add_net_overlap_constraints_between_separate_nets()
        self.add_edge_overlap_constraint()

    def add_flow_constraints(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, cp_model.IntVar]] = {}
        self.edge_flow_value[i] = {edge: self.model.NewIntVar(0, self.K[i], f"edge_flow_value_{i}_{edge}") for edge in self.all_edges[i]}

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges[i]:
            self.model.Add(self.edge_flow_value[i][edge] >= 1).OnlyEnforceIf(self.is_edge_used[i][edge])
            self.model.Add(self.edge_flow_value[i][edge] == 0).OnlyEnforceIf(self.is_edge_used[i][edge].Not())

        # Flow conservation constraints
        for node in self.all_nodes[i]:
            in_flow = sum(self.edge_flow_value[i][edge] for edge in self.all_edges[i] if edge[1] == node)
            out_flow = sum(self.edge_flow_value[i][edge] for edge in self.all_edges[i] if edge[0] == node)

            if node == self.start[i]:
                self.model.Add(out_flow - in_flow == self.K[i])
            elif node in self.goals[i]:
                self.model.Add(out_flow - in_flow == -1)
            else:
                self.model.Add(out_flow - in_flow == 0)
    
    def add_flow_constraints_v2(self, i):
        # Flow conservation constraints
        for node in self.all_nodes[i]:
            in_flow = [self.is_edge_used[i][edge] for edge in self.all_edges[i] if edge[1] == node]
            out_flow = [self.is_edge_used[i][edge] for edge in self.all_edges[i] if edge[0] == node]

            if node == self.start[i]:
                self.model.AddBoolOr(out_flow)
                self.model.AddBoolAnd([e.Not() for e in in_flow])
            elif node in self.goals[i]:
                self.model.Add(sum(in_flow) == 1)
                self.model.AddBoolAnd([e.Not() for e in out_flow])
            else:
                # Create helper variables
                has_in = self.model.NewBoolVar(f"has_in_{i}_{node}")
                has_out = self.model.NewBoolVar(f"has_out_{i}_{node}")

                self.model.AddBoolOr(in_flow).OnlyEnforceIf(has_in)
                self.model.AddBoolAnd([e.Not() for e in in_flow]).OnlyEnforceIf(has_in.Not())

                self.model.AddBoolOr(out_flow).OnlyEnforceIf(has_out)
                self.model.AddBoolAnd([e.Not() for e in out_flow]).OnlyEnforceIf(has_out.Not())

                # Enforce bidirectional use: if in then out, if out then in
                self.model.Add(has_in == has_out)

        # the flow in and flow out must not be cyclic
        max_level = self.WIDTH + self.HEIGHT  # rough upper bound for longest path
        node_level = {}

        for node in self.all_nodes[i]:
            node_level[node] = self.model.NewIntVar(0, max_level, f"level_{i}_{node}")
        
        for edge in self.all_edges[i]:
            u, v, _ = edge
            self.model.Add(node_level[v] > node_level[u]).OnlyEnforceIf(self.is_edge_used[i][edge])

    def add_simple_flow_constraints(self):
        # Flow conservation constraints
        for node in self.all_nodes:
            in_flow_bool_list = [self.is_edge_used[edge] for edge in self.all_edges if edge[1] == node]
            out_flow_bool_list = [self.is_edge_used[edge] for edge in self.all_edges if edge[0] == node]

            if node in self.starts:
                self.model.AddBoolOr(out_flow_bool_list)
                self.model.AddBoolAnd([e.Not() for e in in_flow_bool_list])
            elif node in self.goals:
                self.model.AddBoolOr(in_flow_bool_list)
                self.model.AddBoolAnd([e.Not() for e in out_flow_bool_list])
            else:
                self.model.Add(sum(in_flow_bool_list) == sum(out_flow_bool_list))

                # in_flow_bool = self.model.NewBoolVar(f"in_flow_{node}")
                # out_flow_bool = self.model.NewBoolVar(f"out_flow_{node}")
                # self.model.AddBoolOr(in_flow_bool_list).OnlyEnforceIf(in_flow_bool)
                # self.model.AddBoolAnd([e.Not() for e in in_flow_bool_list]).OnlyEnforceIf(in_flow_bool.Not())
                # self.model.AddBoolOr(out_flow_bool_list).OnlyEnforceIf(out_flow_bool)
                # self.model.AddBoolAnd([e.Not() for e in out_flow_bool_list]).OnlyEnforceIf(out_flow_bool.Not())
                
                # # Enforce bidirectional use: if in then out, if out then in
                # self.model.Add(in_flow_bool == out_flow_bool)


    # def add_directional_constraints(self, i):
    #     for jump_edge in self.jump_edges[i]:
    #         u, v, direction = jump_edge
    #         # A jump from u to v in direction `direction` is only allowed
    #         # if there is incoming flow to u from the same direction.

    #         # Collect all edges (step and jump) that go into `u` from direction `direction`
    #         incoming_edges_in_same_direction = []
    #         for edge in self.all_edges[i]:
    #             _, target, dir_step = edge
    #             if target == u and dir_step == direction:
    #                 incoming_edges_in_same_direction.append(self.is_edge_used[i][edge])

    #         # create a variable that sums up the incoming edges in the same direction
    #         sum_of_incoming_edge_in_same_direction = sum(incoming_edges_in_same_direction)

    #         # Enforce that jump flow is only allowed if incoming flow matches direction
    #         self.model.Add(self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction)

    def add_directional_constraints(self):
        # A jump from u to v in direction `direction` is only allowed
        # if there is incoming flow to u from the same direction.

        for jump_edge in self.jump_edges:
            u, _, direction = jump_edge

            if u in self.starts:
                if direction != (0, 1):
                    # Jump from start node is only allowed in the down direction
                    self.model.Add(self.is_edge_used[jump_edge] == 0)
                continue

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction_bool_list = []
            for edge in self.all_edges:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction_bool_list.append(self.is_edge_used[edge])

            if incoming_edges_in_same_direction_bool_list:
                # Create an OR condition: at least one incoming edge in the same direction
                condition = self.model.NewBoolVar(f"jump_allowed_into_{u}_dir_{direction}")
                self.model.AddBoolOr(incoming_edges_in_same_direction_bool_list).OnlyEnforceIf(condition)
                self.model.AddBoolAnd([e.Not() for e in incoming_edges_in_same_direction_bool_list]).OnlyEnforceIf(condition.Not())

                # Enforce jump only if that condition is true
                self.model.AddImplication(self.is_edge_used[jump_edge], condition)
            else:
                # No incoming edges in that direction => forbid the jump
                self.model.Add(self.is_edge_used[jump_edge] == 0)

    def add_overlap_and_one_jump_constraints(self):
        for node in self.all_nodes:
            # list of all jump edges of this node
            jump_edges_related_to_node = [self.is_edge_used[edge] for edge in self.jump_edges_at_node[node]]

            # the constraint
            self.model.AddAtMostOne(jump_edges_related_to_node + [self.is_step_edge_used_at_node[node]])

    # def add_overlap_constraints_v3(self):
    #     for node in self.all_nodes[0]:
    #         step_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)
    #         jump_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)

    #         for i in range(self.num_nets):
    #             step_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[i][node]]
    #             jump_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[i][node]]

    #         # the constraint
    #         self.model += (
    #             pulp.lpSum(pulp.lpSum(step_edges_of_net[i]) / len(step_edges_of_net[i]) for i in range(self.num_nets)) + 
    #             pulp.lpSum(pulp.lpSum(jump_edges_of_net[i]) for i in range(self.num_nets)) <= 1)

    # def add_overlap_constraints(self, i):
    #     for node in self.all_nodes[i]:
    #         # Constraint: a node cannot be used by both step and jump
    #         self.model += (
    #             self.is_node_used_by_step_edge[i][node] + self.is_node_used_by_jump_edge[i][node] <= 1
    #         )

    # def add_one_jump_constraints(self, i):
    #     # at most one jump edge allowed in each node
    #     for node in self.all_nodes[i]:
    #         # create a variable that sums up the jump edges in the same node
    #         num_of_jump_edges_per_node = pulp.lpSum(self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[i][node])

    #         # Enforce that at most one jump edge is allowed in each node
    #         self.model += num_of_jump_edges_per_node <= 1


    def add_goal_action_constraints(self):
        # # no action is to be taken at the goal nodes for any net
        # for goal in self.goals:
        #     self.model.Add(self.is_node_used_at_node[goal] == 0)
        pass

    def add_net_overlap_constraints(self):
        # no overlap between nets
        for node in self.all_nodes[0]:
            list_of_nets_using_node = []
            for i in range(self.num_nets):
                list_of_nets_using_node.append(self.is_node_used_by_net[i][node])
            
            # constraint: at most one net can use a node
            self.model.AddAtMostOne(list_of_nets_using_node)

    # def add_net_overlap_constraints_between_separate_nets(self):
    #     # no overlap between nets that don't share start node
    #     for node in self.all_nodes:
    #         # for each i and j, used by i means not used by j if they don't have same start
    #         for i in range(self.num_nets):
    #             # list of nets that don't share start node with i
    #             list_of_nets_using_node = [self.is_node_used_by_net[j][node] for j in range(self.num_nets) if self.start[i] != self.start[j]]
    #             self.model.Add(sum(list_of_nets_using_node) == 0).OnlyEnforceIf(self.is_node_used_by_net[i][node])

    #             for j in range(self.num_nets):
    #                 if i != j and self.start[i] != self.start[j]:
    #                     self.model.Add(self.is_node_used_by_net[i][node] + self.is_node_used_by_net[j][node] <= 1)

    def add_net_overlap_constraints_by_edge(self):
        # for each net, for each node, if a edge is used, then no other net can have edge on that node
        for i in range(self.num_nets):
            for node in self.all_nodes[i]:
                # list of all edges of this node
                edges_related_to_node = self.node_related_step_edges[i][node] + self.node_related_jump_edges[i][node]

                # for each edge related node
                for edge in edges_related_to_node:
                    # for each net
                    for j in range(self.num_nets):
                        if j != i:
                            # if edge is used, that implies that no other net can have edge on that node
                            edges_related_to_node_in_other_net = self.node_related_step_edges[j][node] + self.node_related_jump_edges[j][node]
                            
                            self.model.AddBoolAnd([self.is_edge_used[j][edge].Not() for edge in edges_related_to_node_in_other_net]).OnlyEnforceIf(self.is_edge_used[i][edge])

    def add_edge_overlap_constraint(self):
        # node used by step
        # node used by jump_1 in one of 4 direction
        # node used by jump_2 in one of 4 direction
        # node used by jump_3 in one of 4 direction
        # node used by jump_4 in one of 4 direction
        # for each node, it can only be used by one type of edge, this is true no matter what nets

        # for each node
        for node in self.all_nodes:

            # node used by any one jump from any net
            jump_edge_bools_list = []

            # for each different jump edge
            for jump_edge in self.jump_edges_at_node[node]:
                this_jump_edge_used_at_this_node = self.model.NewBoolVar(f"jump_edge_at_this_node_{node}")
                this_jump_edges_bool_at_this_node = []
                for i in range(self.num_nets):
                    this_jump_edges_bool_at_this_node += [self.is_edge_used[edge] for edge in self.jump_edges_at_node[node] if edge == jump_edge]
                
                self.model.AddBoolOr(this_jump_edges_bool_at_this_node).OnlyEnforceIf(this_jump_edge_used_at_this_node)
                self.model.AddBoolAnd([edge.Not() for edge in this_jump_edges_bool_at_this_node]).OnlyEnforceIf(this_jump_edge_used_at_this_node.Not())

                jump_edge_bools_list.append(this_jump_edge_used_at_this_node)
            
            # add the constraint that at most one type of edge (where step edges of different direction are considered one type) can be used at this node
            self.model.AddAtMostOne(jump_edge_bools_list + [self.is_step_edge_used_at_node[node]])

    def add_symmetry_constraints(self):
        for edge in self.all_edges:
            u, v, d = edge
            sym_ux = self.WIDTH - u[0] - 1
            sym_suy = u[1]
            sym_svx = self.WIDTH - v[0] - 1
            sym_svy = v[1]
            sym_u = (sym_ux, sym_suy)
            sym_v = (sym_svx, sym_svy)
            sym_d = (-d[0], d[1])
            if ((sym_u, sym_v, sym_d) in self.all_edges):
                self.model.Add(self.is_edge_used[edge] == self.is_edge_used[(sym_u, sym_v, sym_d)])

    def solve(self):
        self.solver.parameters.log_search_progress = True
        # self.solver.parameters.num_search_workers = 8  # for parallelism
        status = self.solver.Solve(self.model)

        # Print optimization time
        print(f"Solver status: {self.solver.StatusName(status)}")
        print(f"Optimization wall time: {self.solver.WallTime():.3f} seconds")
        print(f"Objective value: {self.solver.ObjectiveValue()}")

    def plot(self):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']    
        # for i in range(self.num_nets):
        # color = colors[i % len(colors)]
        color = 'red'

        used_step_edges = [e for e in self.step_edges if self.solver.Value(self.is_edge_used[e]) == 1]
        used_jump_edges = [e for e in self.jump_edges if self.solver.Value(self.is_edge_used[e]) == 1]


        # Plot start and goal
        for i in range(self.num_nets):
            start = self.starts[i]
            sx, sy = start
            plt.scatter(sx + offset, sy + offset, c=color, marker='s', s=120, edgecolors='black', label='Start', zorder = 0)

            gx, gy = self.goals[i]
            plt.scatter(gx + offset, gy + offset, c=color, marker='s', s=120, edgecolors='black', zorder = 0)
            plt.scatter(gx + offset, gy + offset, c=color, marker='o', s=50, edgecolors='black', zorder = 2)
        

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
        ((5, 0), (1, 6)),
        # ((6, 0), [(3, 6)]),
        # ((6, 0), [(9, 6), (11, 6), (13, 6), (15, 6)]),
        # ((7, 0), [(17, 6), (19, 6), (21, 6), (23, 6)]),
        # ((8, 0), [(25, 6), (27, 6), (29, 6), (31, 6)]),
        # ((25, 0), [(2, 6), (4, 6), (6, 6), (8, 6)]),
        # ((26, 0), [(10, 6), (12, 6), (14, 6), (16, 6)]),
        # ((27, 0), [(18, 6), (20, 6), (22, 6), (24, 6)]),
        # ((28, 0), [(26, 6), (28, 6), (30, 6), (32, 6)]),
        ]
    router = DirectionalJumpRouter(width=34, height=7, nets=nets, jump_distances= [1, 2, 3, 4], timelimit = 120)