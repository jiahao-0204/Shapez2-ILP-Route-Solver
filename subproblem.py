from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List, Set
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple


STEP_COST = 1
JUMP_COST = 2
STARTING_PAD = 0
LANDING_PAD = 1
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)] # right, up, left, down

PAD_TYPE = int
Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Edge = Tuple[Node, Node, Direction] # start, end, direciton

class SubProblem:
    def __init__(self):
        
        
        self.WIDTH = 16
        self.HEIGHT = 16
        self.jump_distances = [1, 2, 3, 4]
        self.timelimit = -1
        self.option = 1

        # problem parameters
        self.component_source_amount = 1
        self.component_sink_amount = 1

        self.num_nets = 3 # start to component, component to goal

        self.preplacement_list = []
        self.preplacement_list.append(((4, 3), (1, 0), (0, 1)))
        self.preplacement_list.append(((4, 7), (1, 0), (0, -1)))
        self.preplacement_list.append(((4, 9), (1, 0), (0, 1)))
        # self.preplacement_list.append(((4, 13), (1, 0), (0, -1)))

        self.preplacement_list.append(((6, 3), (-1, 0), (0, 1)))
        self.preplacement_list.append(((6, 7), (-1, 0), (0, -1)))
        self.preplacement_list.append(((6, 9), (-1, 0), (0, 1)))
        # self.preplacement_list.append(((6, 13), (-1, 0), (0, -1)))

        self.preplacement_list.append(((9, 3), (1, 0), (0, 1)))
        self.preplacement_list.append(((9, 7), (1, 0), (0, -1)))
        self.preplacement_list.append(((9, 9), (1, 0), (0, 1)))
        # self.preplacement_list.append(((9, 13), (1, 0), (0, -1)))

        self.preplacement_list.append(((11, 3), (-1, 0), (0, 1)))
        self.preplacement_list.append(((11, 7), (-1, 0), (0, -1)))
        self.preplacement_list.append(((11, 9), (-1, 0), (0, 1)))
        # self.preplacement_list.append(((11, 13), (-1, 0), (0, -1)))


        self.flow_cap = 4
        self.start_amount = 4
        self.goal_amount = 4
        self.edge_priority = 50
        self.flow_priority = 25

        # blocked tile is the border of the map
        self.border = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
        self.corner = [(1, 1), (self.WIDTH-2, 1), (1, self.HEIGHT-2), (self.WIDTH-2, self.HEIGHT-2)]

        self.port_location = []
        self.port_location += [(6, 1), (7, 1), (8, 1), (9, 1)]
        self.port_location += [(6, self.HEIGHT-2), (7, self.HEIGHT-2), (8, self.HEIGHT-2), (9, self.HEIGHT-2)]
        self.port_location += [(1, 6), (1, 7), (1, 8), (1, 9)]
        self.port_location += [(self.WIDTH-2, 6), (self.WIDTH-2, 7), (self.WIDTH-2, 8), (self.WIDTH-2, 9)]

        self.port_center_location = []
        self.port_center_location += [(7, 1), (8, 1)]
        self.port_center_location += [(7, self.HEIGHT-2), (8, self.HEIGHT-2)]
        self.port_center_location += [(1, 7), (1, 8)]
        self.port_center_location += [(self.WIDTH-2, 7), (self.WIDTH-2, 8)]

        self.blocked_tiles = self.border.copy()
        
        remove_from_blocked_tiles = [] 
        remove_from_blocked_tiles += [(6, 0), (7, 0), (8, 0), (9, 0)]
        remove_from_blocked_tiles += [(6, self.HEIGHT-1), (7, self.HEIGHT-1), (8, self.HEIGHT-1), (9, self.HEIGHT-1)]
        # remove_from_blocked_tiles += [(26, self.HEIGHT-1), (27, self.HEIGHT-1), (28, self.HEIGHT-1), (29, self.HEIGHT-1)]
        remove_from_blocked_tiles += [(0, 6), (0, 7), (0, 8), (0, 9)]
        remove_from_blocked_tiles += [(self.WIDTH-1, 6), (self.WIDTH-1, 7), (self.WIDTH-1, 8), (self.WIDTH-1, 9)]

        for tile in remove_from_blocked_tiles:
            self.blocked_tiles.remove(tile)
        # self.blocked_tiles += [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]

        # all nodes
        self.all_nodes: List[Node] = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if (x, y) not in self.blocked_tiles:
                    self.all_nodes.append((x, y))

        self.all_edges: List[Edge] = []
        self.step_edges: List[Edge] = []
        self.jump_edges: List[Edge] = []
        self.node_related_belt_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_starting_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_landing_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)
        for node in self.all_nodes:
            x, y = node
            for dx, dy in DIRECTIONS:

                # Step edge
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.all_nodes:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.step_edges.append(edge)
                    self.node_related_belt_edges[node].append(edge)
                
                for jump_distance in self.jump_distances:
                    nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
                    jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
                    pad_node = (jx, jy)
                    if (nx, ny) in self.all_nodes:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges.append(edge)
                        self.jump_edges.append(edge)
                        self.node_related_starting_pad_edges[node].append(edge)
                        self.node_related_landing_pad_edges[pad_node].append(edge)

        # start and goals
        self.starts: List[Tuple[Node, Direction]] = []
        self.starts.append(((7, 0), (0, 1)))
        self.starts.append(((8, 0), (0, 1)))
        self.starts.append(((9, 0), (0, 1)))

        self.goals1: List[Tuple[Node, Direction]] = []
        self.goals1.append(((0, 7), (-1, 0)))
        self.goals1.append(((0, 8), (-1, 0)))
        self.goals1.append(((0, 9), (-1, 0)))

        self.goals2: List[Tuple[Node, Direction]] = []
        self.goals2.append(((7, 15), (0, 1)))
        self.goals2.append(((8, 15), (0, 1)))
        self.goals2.append(((9, 15), (0, 1)))

        self.net_sources: Dict[int, List[Node]] = defaultdict(list)
        self.net_sinks: Dict[int, List[Node]] = defaultdict(list)
        self.net_sources[0] = [node for node, _ in self.starts]
        self.net_sinks[1] = [node for node, _ in self.goals1]
        self.net_sinks[2] = [node for node, _ in self.goals2]

        feasible, cost, is_edge_used = self.solve_subproblem(self.preplacement_list, self.starts, self.goals1, self.goals2)

        self.plot(is_edge_used, self.preplacement_list)

    def solve_subproblem(self, cutters, starts, goals1, goals2):
        # set up model parameters
        sub_model = Model("subproblem")
        if self.timelimit != -1:
            sub_model.Params.TimeLimit = self.timelimit
        sub_model.Params.MIPFocus = 1
        sub_model.Params.Presolve = 2
        # sub_model.Params.OutputFlag = 0  # silent

        self.is_edge_used: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.node_in_flow_edges: Dict[int, Dict[Node, List[Edge]]] = defaultdict(lambda: defaultdict(list))
        self.node_out_flow_edges: Dict[int, Dict[Node, List[Edge]]] = defaultdict(lambda: defaultdict(list))
        self.node_in_flow_value_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.node_out_flow_value_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.is_node_used_by_belt: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))

        # edge and flow
        for i in range(self.num_nets):
            for edge in self.all_edges:
                # edge flow value
                self.edge_flow_value[i][edge] = sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap)
                self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

                # is edge used
                self.is_edge_used[i][edge] = sub_model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY)
                self.is_edge_used[i][edge].setAttr("BranchPriority", self.edge_priority)

                # in flow and out flow values
                self.node_in_flow_value_expr[i][edge[1]].addTerms(1, self.edge_flow_value[i][edge])
                self.node_out_flow_value_expr[i][edge[0]].addTerms(1, self.edge_flow_value[i][edge])

                # node in and out flow edges
                self.node_in_flow_edges[i][edge[1]].append(edge)
                self.node_out_flow_edges[i][edge[0]].append(edge)
        self.compute_is_edge_used(sub_model)
        self.compute_is_node_used_by_belt(sub_model)
        self.add_flow_max_value_constraints(sub_model)
        self.add_belt_pad_net_overlap_constraints(sub_model)
        self.add_pad_direction_constraints(sub_model)
        
        # general
        self.add_objective(sub_model)
        
        # add start and goal
        self.add_start_edge_constraints(sub_model, starts)
        self.add_goal_edge_constraints(sub_model, goals1)
        self.add_goal_edge_constraints(sub_model, goals2)

        # add cutter
        self.add_cutter_edge_constraints(sub_model, cutters)
        self.add_cutter_net(sub_model, cutters, starts, goals1, goals2)

        # solve
        self.solve(sub_model)

        # get solution
        if sub_model.Status == GRB.INFEASIBLE:
            return False, None, None
        else:
            is_edge_used = {}
            for i in range(self.num_nets):
                is_edge_used[i] = {edge: sub_model.getVarByName(f"edge_{i}_{edge}").X for edge in self.all_edges}
                self.is_edge_used[i] = is_edge_used[i]
            return True, sub_model.ObjVal, is_edge_used

        # # Plot
        # self.plot()

    def add_objective(self, sub_model):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        sub_model.setObjective(quicksum(step_cost_list + jump_cost_list))

    def compute_is_node_used_by_belt(self, sub_model):
        # compute is node used by belt edge
        for i in range(self.num_nets):
            for node in self.all_nodes:
                node_belt_edges_bool_list = [self.is_edge_used[i][edge] for edge in self.node_related_belt_edges[node]]
                self.is_node_used_by_belt[i][node] = sub_model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)

                sub_model.addGenConstrOr(self.is_node_used_by_belt[i][node], node_belt_edges_bool_list)

    def compute_is_edge_used(self, sub_model):
        # compute is edge used
        for i in range(self.num_nets):
            for edge in self.all_edges:
                # constraint
                sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
                sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

    def add_flow_max_value_constraints(self, sub_model):
        # max flow at each node
        for i in range(self.num_nets):
            for node in self.all_nodes:
                sub_model.addConstr(self.node_in_flow_value_expr[i][node] <= self.flow_cap)
                sub_model.addConstr(self.node_out_flow_value_expr[i][node] <= self.flow_cap)

    def add_belt_pad_net_overlap_constraints(self, sub_model):
        # between belts / pads in different nets
        for node in self.all_nodes:
            list_of_things_using_node = []
            for i in range(self.num_nets):
                list_of_things_using_node += [self.is_node_used_by_belt[i][node]]
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_starting_pad_edges[node]]
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_landing_pad_edges[node]]
            
            # constraint: at most one thing can use a node
            sub_model.addConstr(quicksum(list_of_things_using_node) <= 1)
    
    def add_start_edge_constraints(self, sub_model, starts: List[Tuple[Node, Direction]]):
        for node, direction in starts:
            # null node
            null_node = node
            self.add_null_node_constraints(sub_model, null_node)
            # source node
            source_node = (node[0] + direction[0], node[1] + direction[1])
            self.add_source_node_constraints(sub_model, source_node, direction)
    
    def add_goal_edge_constraints(self, sub_model, goals: List[Tuple[Node, Direction]]):
        for node, direction in goals:
            self.add_sink_node_constraints(sub_model, node, direction)

    def add_pad_direction_constraints(self, sub_model):
        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for i in range(self.num_nets):
            for edge in self.all_edges:
                u, v, direction = edge
                
                # no landing pad if edge is used
                for jump_edge in self.node_related_landing_pad_edges[v]:
                    sub_model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

                # no starting pad at wrong direction if edge is used
                for jump_edge in self.node_related_starting_pad_edges[v]:
                    u2, v2, jump_direction = jump_edge
                    # skip if correct direction
                    if jump_direction == direction:
                        continue
                    sub_model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

    def add_cutter_net(self, sub_model, cutters, starts, goals1, goals2):
        # net 0: start -> componenent sink
        # net 1: component source -> goal
        # net 2: component secondary source -> goal
        s0 = [(node[0] + direction[0], node[1] + direction[1]) for node, direction in starts]
        k0 = []
        s1 = []
        k1 = [node for node, _ in goals1]
        s2 = []
        k2 = [node for node, _ in goals2]

        for cutter in cutters:
            sink, direction, secondary_direction = cutter
            primary_source = (sink[0] + direction[0], sink[1] + direction[1])
            secondary_source = (primary_source[0] + secondary_direction[0], primary_source[1] + secondary_direction[1])
            k0 += [sink]
            s1 += [primary_source]
            s2 += [secondary_source]

        s0_amount = [self.start_amount] * len(s0)
        k0_amount = [self.component_sink_amount] * len(k0)
        s1_amount = [self.component_source_amount] * len(s1)
        k1_amount = [self.goal_amount] * len(k1)
        s2_amount = [self.component_source_amount] * len(s2)
        k2_amount = [self.goal_amount] * len(k2)

        self.add_net(sub_model, 0, s0, s0_amount, k0, k0_amount)
        self.add_net(sub_model, 1, s1, s1_amount, k1, k1_amount)
        self.add_net(sub_model, 2, s2, s2_amount, k2, k2_amount)

    # within one net, flow can split and merge
    def add_net(self, sub_model, i, sources, source_amounts, sinks, sink_amounts):
        for node in self.all_nodes:
            in_flow = self.node_in_flow_value_expr[i][node]
            out_flow = self.node_out_flow_value_expr[i][node]

            if node in sources:
                source_count = sources.count(node)
                sub_model.addConstr(out_flow - in_flow == source_amounts[sources.index(node)] * source_count)
            elif node in sinks:
                sink_count = sinks.count(node)
                sub_model.addConstr(in_flow - out_flow == sink_amounts[sinks.index(node)] * sink_count)
            else:
                sub_model.addConstr(in_flow - out_flow == 0)

    def add_cutter_edge_constraints(self, sub_model, cutters):
        # split into lists
        primary_components: List[Tuple[Node, Direction]] = []
        secondary_components: List[Tuple[Node, Direction]] = []
        primary_sources: List[Tuple[Node, Direction]] = []
        secondary_sources: List[Tuple[Node, Direction]] = []
        for cutter in cutters:
            primary_component, direction, secondary_direction = cutter
            secondary_component = (primary_component[0] + secondary_direction[0], primary_component[1] + secondary_direction[1])
            primary_source = (primary_component[0] + direction[0], primary_component[1] + direction[1])
            secondary_source = (secondary_component[0] + direction[0], secondary_component[1] + direction[1])
            input_location = (primary_component[0] - direction[0], primary_component[1] - direction[1])
            
            primary_components.append((primary_component, direction))
            secondary_components.append((secondary_component, secondary_direction))
            primary_sources.append((primary_source, direction))
            secondary_sources.append((secondary_source, direction))

        # primary component nodes
        for node, direction in primary_components:
            self.add_sink_node_constraints(sub_model, node, direction)
                
        # secondary component nodes
        for node, direction in secondary_components:
            self.add_null_node_constraints(sub_model, node)
                
        # primary source nodes
        for node, direction in primary_sources:
            self.add_source_node_constraints(sub_model, node, direction)
        
        # secondary source nodes
        for node, direction in secondary_sources:
            self.add_source_node_constraints(sub_model, node, direction)
    
    def add_source_node_constraints(self, sub_model, node:Node, direction:Direction):
        for i in range(self.num_nets):
            # inflow: except in opposite direction
            for edge in self.node_in_flow_edges[i][node]:
                if edge[2] == (-direction[0], -direction[1]):
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # outflow: except in opposite direction
            for edge in self.node_out_flow_edges[i][node]:
                if edge[2] == (-direction[0], -direction[1]):
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: only in direction
            for edge in self.node_related_starting_pad_edges[node]:
                if edge[2] != direction:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # landing pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

    def add_sink_node_constraints(self, sub_model, node:Node, direction:Direction):
        for i in range(self.num_nets):
            # in flow: only in direction
            for edge in self.node_in_flow_edges[i][node]:
                if edge[2] != direction:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            for edge in self.node_out_flow_edges[i][node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: no
            for edge in self.node_related_starting_pad_edges[node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # land pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)
            
            # ------ for input location ------
            input_node = (node[0] - direction[0], node[1] - direction[1])
            
            # in flow: except from node direction (but is covered in the above)
            pass

            # outflow: all
            pass

            # start pad: no
            for edge in self.node_related_starting_pad_edges[input_node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)
            
            # land pad: only in direction
            for edge in self.node_related_landing_pad_edges[input_node]:
                if edge[2] != direction:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

    def add_null_node_constraints(self, sub_model, node:Node):
        for i in range(self.num_nets):
            # in flow: no
            for edge in self.node_in_flow_edges[i][node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            for edge in self.node_out_flow_edges[i][node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: no
            for edge in self.node_related_starting_pad_edges[node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # land pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

    def solve(self, sub_model):
        if self.timelimit != -1:
            sub_model.setParam('TimeLimit', self.timelimit)
        sub_model.setParam('MIPFocus', self.option)
        sub_model.setParam('Presolve', 2)
        sub_model.setParam('Heuristics', 0.5)
        sub_model.update()
        sub_model.optimize()
        
        # # Copy and apply feasibility relaxation
        # relaxed_model = sub_model.copy()
        # relaxed_model.feasRelaxS(0, False, False, True)
        # relaxed_model.optimize()

    def plot(self, sub_problem_is_edge_used, used_components):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5

        # draw blocked tiles
        for (x, y) in self.blocked_tiles:
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='none', hatch='////'))
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=2))
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', linewidth=2))


        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']    
        for i in range(self.num_nets):
            color = colors[i % len(colors)]

            used_step_edges = [e for e in self.step_edges if sub_problem_is_edge_used[i][e] == 1]
            used_jump_edges = [e for e in self.jump_edges if sub_problem_is_edge_used[i][e] == 1]

            # Plot start and goal
            for start in self.net_sources[i]:
                sx, sy = start
                plt.scatter(sx + offset, sy + offset, c=color, marker='s', s=120, edgecolors='black', label='Start', zorder = 0)
            for goal in self.net_sinks[i]:
                gx, gy = goal
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
        
        # draw components
        for component in used_components:
            (x, y), (dx, dy), (dx2, dy2) = component        # two‑cell component
            nx, ny = x + dx, y + dy
            x2, y2 = x + dx2, y + dy2                       # second node
            nx2, ny2 = x2 + dx, y2 + dy

            margin = 0.2
            ll_x = min(x, x2) + margin
            ll_y = min(y, y2) + margin
            width  = abs(x2 - x) + 1 - 2 * margin       # +1 because each node is 1×1
            height = abs(y2 - y) + 1 - 2 * margin

            rect = plt.Rectangle((ll_x, ll_y), width, height, facecolor='grey', edgecolor='black', linewidth=1.2, zorder=1)
            ax.add_patch(rect)
            d = (dx, dy)
            if d == (0, 1):
                marker = '^'
            elif d == (0, -1):
                marker = 'v'
            elif d == (1, 0):
                marker = '>'
            elif d == (-1, 0):
                marker = '<'
            ax.scatter(x + offset, y + offset, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
            ax.scatter(x2 + offset, y2 + offset, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
            ax.plot([x + offset, nx + offset], [y + offset, ny + offset], c='black', zorder=0)
            ax.plot([x2 + offset, nx2 + offset], [y2 + offset, ny2 + offset], c='black', zorder=0)

        plt.title("Shapez2: Routing using Integer Linear Programming (ILP) -- Jiahao")

        # custom legend
        handle_start = Line2D([], [], marker='s', color='grey', markersize=9, markeredgecolor='black', linestyle='None', label='Start/Goal')
        handle_jump_pad = Line2D([], [], marker='^', color='grey', markersize=8, markeredgecolor='black', linestyle='None', label='Jump Pad')
        handle_belt = Line2D([], [], marker='o', color='grey', markersize=7, markeredgecolor='black', linestyle='None', label='Belt')
        # handle_component_square = Line2D([], [], marker='s', color='grey', markersize=14, markeredgecolor='black', linestyle='None')
        # handle_component_circle = Line2D([], [], marker='o', color='grey', markersize=13, markeredgecolor='black', linestyle='None')
        # handle_component = (handle_component_square, handle_component_circle)
        legend_handles = [handle_start, handle_jump_pad, handle_belt]
        legend_labels  = ['Start/Goal', 'Jump Pad', 'Belt']
        ax.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc='upper right')

        # show
        plt.show()

if __name__ == "__main__":
    router = SubProblem()