from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List, Set
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# constants
STEP_COST = 1
JUMP_COST = 2
EDGE_PRIORITY = 50
FLOW_PRIORITY = 25

# flow 
FLOW_CAP = 4
IO_AMOUNT = 4
CUTTER_AMOUNT = 1

# pad types
PAD_TYPE = int
STARTING_PAD = 0
LANDING_PAD = 1

# directions
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)] # right, up, left, down

# options
MIPFOCOUS_TYPE = int
MIPFOCUS_BALANCED = 0
MIPFOCUS_FEASIBILITY = 1
MIPFOCUS_OPTIMALITY = 2
MIPFOCUS_BOUND = 3
NO_TIME_LIMIT = -1

# solver settings
PRESOLVE = 2
HEURISTICS = 0.5

# types
Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Edge = Tuple[Node, Node, Direction] # start, end, direciton

class SubProblem:
    def __init__(self):
        pass

    def route_cutters(self, width, height, cutters, starts, goals1, goals2, jump_distances, timelimit, option: MIPFOCOUS_TYPE):
        # general settings
        num_nets = 3
        self.initialize_board(width, height, jump_distances, num_nets)

        # cutter settings
        self.add_start_edge_constraints(starts)
        self.add_goal_edge_constraints(goals1 + goals2)

        io_tiles = [node for node, _ in starts + goals1 + goals2]
        self.add_border_edge_constraints(io_tiles)

        self.add_cutter_edge_constraints(cutters)
        self.add_cutter_net(cutters, starts, goals1, goals2)

        # solve
        used_edge = self.solve(timelimit, option)

        # draw solution
        if used_edge is not None:
            self.draw(cutters, used_edge)

    def initialize_board(self, width, height, jump_distances, num_nets):
        # add model
        self.model = Model("subproblem")

        # board
        self.WIDTH = width
        self.HEIGHT = height
        self.all_nodes: List[Node] = [(x, y) for x in range(self.WIDTH) for y in range(self.HEIGHT)]
        self.num_nets = num_nets

        # common variables across all nets
        self.all_edges: List[Edge] = []
        self.step_edges: List[Edge] = []
        self.jump_edges: List[Edge] = []
        self.node_related_belt_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_starting_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_landing_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_in_flow_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_out_flow_edges: Dict[Node, List[Edge]] = defaultdict(list)
        for node in self.all_nodes:
            for dx, dy in DIRECTIONS:
                x, y = node

                # step edges
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.all_nodes:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.step_edges.append(edge)
                    self.node_related_belt_edges[node].append(edge)
                    self.node_out_flow_edges[node].append(edge)
                    self.node_in_flow_edges[(nx, ny)].append(edge)
                
                # jump edges
                for jump_distance in jump_distances:
                    nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
                    jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
                    pad_node = (jx, jy)
                    if (nx, ny) in self.all_nodes:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges.append(edge)
                        self.jump_edges.append(edge)
                        self.node_related_starting_pad_edges[node].append(edge)
                        self.node_related_landing_pad_edges[pad_node].append(edge)
                        self.node_out_flow_edges[node].append(edge)
                        self.node_in_flow_edges[(nx, ny)].append(edge)

        # net specific variables
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.is_edge_used: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.node_in_flow_value_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.node_out_flow_value_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.is_node_used_by_belt: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for edge in self.all_edges:
                # edge flow value
                self.edge_flow_value[i][edge] = self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=FLOW_CAP)
                self.edge_flow_value[i][edge].setAttr("BranchPriority", FLOW_PRIORITY)

                # is edge used (dynamic variable)
                self.is_edge_used[i][edge] = self.model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY)
                self.is_edge_used[i][edge].setAttr("BranchPriority", EDGE_PRIORITY)
                self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
                self.model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

                # in flow and out flow values
                self.node_in_flow_value_expr[i][edge[1]].addTerms(1, self.edge_flow_value[i][edge])
                self.node_out_flow_value_expr[i][edge[0]].addTerms(1, self.edge_flow_value[i][edge])
        for i in range(self.num_nets):
            for node in self.all_nodes:
                # is node used by belt (dynamic variable)
                node_belt_edges_bool_list = [self.is_edge_used[i][edge] for edge in self.node_related_belt_edges[node]]
                self.is_node_used_by_belt[i][node] = self.model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)

                self.model.addGenConstrOr(self.is_node_used_by_belt[i][node], node_belt_edges_bool_list)
        
        # constraints
        self.add_flow_max_value_constraints()
        self.add_belt_pad_net_overlap_constraints()
        self.add_pad_direction_constraints()

    def add_flow_max_value_constraints(self):
        # max flow at each node
        for i in range(self.num_nets):
            for node in self.all_nodes:
                self.model.addConstr(self.node_in_flow_value_expr[i][node] <= FLOW_CAP)
                self.model.addConstr(self.node_out_flow_value_expr[i][node] <= FLOW_CAP)

    def add_belt_pad_net_overlap_constraints(self):
        # between belts / pads in different nets
        for node in self.all_nodes:
            list_of_things_using_node = []
            for i in range(self.num_nets):
                list_of_things_using_node += [self.is_node_used_by_belt[i][node]]
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_starting_pad_edges[node]]
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_landing_pad_edges[node]]
            
            # constraint: at most one thing can use a node
            self.model.addConstr(quicksum(list_of_things_using_node) <= 1)
    
    def add_start_edge_constraints(self, starts: List[Tuple[Node, Direction]]):
        # add null and source nodes
        for node, direction in starts:
            # null node
            null_node = node
            self.add_null_node_constraints(null_node)

            # source node
            source_node = (node[0] + direction[0], node[1] + direction[1])
            self.add_source_node_constraints(source_node, direction)
    
    def add_goal_edge_constraints(self, goals: List[Tuple[Node, Direction]]):
        # add as sink node
        for node, direction in goals:
            self.add_sink_node_constraints(node, direction)

    def add_border_edge_constraints(self, io_tiles):
        # border tiles excluding io tiles
        self.border = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
        for tile in io_tiles:
            self.border.remove(tile)
        
        # add as null node
        for node in self.border:
            self.add_null_node_constraints(node)

    def add_pad_direction_constraints(self):
        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for i in range(self.num_nets):
            for edge in self.all_edges:
                u, v, direction = edge
                
                # no landing pad if edge is used
                for jump_edge in self.node_related_landing_pad_edges[v]:
                    self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

                # no starting pad at wrong direction if edge is used
                for jump_edge in self.node_related_starting_pad_edges[v]:
                    u2, v2, jump_direction = jump_edge
                    # skip if correct direction
                    if jump_direction == direction:
                        continue
                    self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

    def add_cutter_net(self, cutters, starts, goals1, goals2):
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

        s0_amount = [IO_AMOUNT] * len(s0)
        k0_amount = [CUTTER_AMOUNT] * len(k0)
        s1_amount = [CUTTER_AMOUNT] * len(s1)
        k1_amount = [IO_AMOUNT] * len(k1)
        s2_amount = [CUTTER_AMOUNT] * len(s2)
        k2_amount = [IO_AMOUNT] * len(k2)

        self.net_sources: Dict[int, List[Node]] = defaultdict(list)
        self.net_sinks: Dict[int, List[Node]] = defaultdict(list)
        self.net_sources[0] = [node for node, _ in starts]
        self.net_sinks[1] = [node for node, _ in goals1]
        self.net_sinks[2] = [node for node, _ in goals2]

        self.add_net(0, s0, s0_amount, k0, k0_amount)
        self.add_net(1, s1, s1_amount, k1, k1_amount)
        self.add_net(2, s2, s2_amount, k2, k2_amount)

    # within one net, flow can split and merge
    def add_net(self, i, sources, source_amounts, sinks, sink_amounts):
        for node in self.all_nodes:
            in_flow = self.node_in_flow_value_expr[i][node]
            out_flow = self.node_out_flow_value_expr[i][node]

            if node in sources:
                source_count = sources.count(node)
                self.model.addConstr(out_flow - in_flow == source_amounts[sources.index(node)] * source_count)
            elif node in sinks:
                sink_count = sinks.count(node)
                self.model.addConstr(in_flow - out_flow == sink_amounts[sinks.index(node)] * sink_count)
            else:
                self.model.addConstr(in_flow - out_flow == 0)

    def add_cutter_edge_constraints(self, cutters):
        for cutter in cutters:
            primary_component, direction, secondary_direction = cutter
            self.add_sink_node_constraints(primary_component, direction)
            
            secondary_component = (primary_component[0] + secondary_direction[0], primary_component[1] + secondary_direction[1])
            self.add_null_node_constraints(secondary_component)
            
            primary_source = (primary_component[0] + direction[0], primary_component[1] + direction[1])
            self.add_source_node_constraints(primary_source, direction)
            
            secondary_source = (secondary_component[0] + direction[0], secondary_component[1] + direction[1])
            self.add_source_node_constraints(secondary_source, direction)
    
    def add_source_node_constraints(self, node:Node, direction:Direction):
        for i in range(self.num_nets):
            # inflow: except in opposite direction
            for edge in self.node_in_flow_edges[node]:
                if edge[2] == (-direction[0], -direction[1]):
                    self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # outflow: except in opposite direction
            for edge in self.node_out_flow_edges[node]:
                if edge[2] == (-direction[0], -direction[1]):
                    self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: only in direction
            for edge in self.node_related_starting_pad_edges[node]:
                if edge[2] != direction:
                    self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # landing pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

    def add_sink_node_constraints(self, node:Node, direction:Direction):
        for i in range(self.num_nets):
            # in flow: only in direction
            for edge in self.node_in_flow_edges[node]:
                if edge[2] != direction:
                    self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            for edge in self.node_out_flow_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: no
            for edge in self.node_related_starting_pad_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # land pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)
            
            # ------ for input location ------
            input_node = (node[0] - direction[0], node[1] - direction[1])
            
            # in flow: except from node direction (but is covered in the above)
            pass

            # outflow: all
            pass

            # start pad: no
            for edge in self.node_related_starting_pad_edges[input_node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)
            
            # land pad: only in direction
            for edge in self.node_related_landing_pad_edges[input_node]:
                if edge[2] != direction:
                    self.model.addConstr(self.is_edge_used[i][edge] == 0)

    def add_null_node_constraints(self, node:Node):
        for i in range(self.num_nets):
            # in flow: no
            for edge in self.node_in_flow_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            for edge in self.node_out_flow_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # start pad: no
            for edge in self.node_related_starting_pad_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

            # land pad: no
            for edge in self.node_related_landing_pad_edges[node]:
                self.model.addConstr(self.is_edge_used[i][edge] == 0)

    def solve(self, timelimit, option):
        # objective
        self.add_objective()

        # settings
        if timelimit != -1:
            self.model.Params.TimeLimit = timelimit
        self.model.Params.MIPFocus = option
        self.model.Params.Presolve = PRESOLVE

        # solve
        self.model.optimize()

        # return solution
        if self.model.SolCount > 0:
            used_edge = {i: [edge for edge in self.all_edges if self.is_edge_used[i][edge].X > 0.5] for i in range(self.num_nets)}
            return used_edge
        else:
            return None

    def add_objective(self):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        self.model.setObjective(quicksum(step_cost_list + jump_cost_list))

    def draw(self, cutters, used_edge):
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5

        # draw border tiles
        for (x, y) in self.border:
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='none', hatch='////'))
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=2))
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', linewidth=2))

        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']    
        for i in range(self.num_nets):
            color = colors[i % len(colors)]

            used_step_edges = [edge for edge in self.step_edges if edge in used_edge[i]]
            used_jump_edges = [edge for edge in self.jump_edges if edge in used_edge[i]]

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
        for cutter in cutters:
            (x, y), (dx, dy), (dx2, dy2) = cutter        # two‑cell component
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

    def draw_components(self, cutter_list):
        # ax = self.ax
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.clear()
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        offset = 0.5

        # draw border tiles
        for (x, y) in self.border:
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='none', hatch='////'))
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=2))
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', linewidth=2))

        # draw components
        used_components = cutter_list
        for component in used_components:
            (x, y), (dx, dy), (dx2, dy2) = component        # two‑cell component
            nx, ny = x + dx, y + dy
            x2, y2 = x + dx2, y + dy2                       # second node
            nx2, ny2 = x2 + dx, y2 + dy

            ix, iy = x - dx, y - dy

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
            ax.plot([ix + offset, x + offset], [iy + offset, y + offset], c='black', zorder=0)
            ax.plot([x + offset, nx + offset], [y + offset, ny + offset], c='black', zorder=0)
            ax.plot([x2 + offset, nx2 + offset], [y2 + offset, ny2 + offset], c='black', zorder=0)

        plt.title("Shapez2: Routing using Integer Linear Programming (ILP) -- Jiahao")

        # plt.draw()
        plt.show()

if __name__ == "__main__":
    # problem definition
    width = 16
    height = 16
    jump_distances = [1, 2, 3, 4]
    time_limit = NO_TIME_LIMIT
    option = MIPFOCUS_BALANCED
    # option = MIPFOCUS_BOUND

    starts: List[Tuple[Node, Direction]] = []
    starts.append(((7, 0), (0, 1)))
    starts.append(((8, 0), (0, 1)))
    starts.append(((9, 0), (0, 1)))

    goals1: List[Tuple[Node, Direction]] = []
    goals1.append(((0, 7), (-1, 0)))
    goals1.append(((0, 8), (-1, 0)))
    goals1.append(((0, 9), (-1, 0)))

    goals2: List[Tuple[Node, Direction]] = []
    goals2.append(((7, 15), (0, 1)))
    goals2.append(((8, 15), (0, 1)))
    goals2.append(((9, 15), (0, 1)))

    cutter_list: List[Tuple[Node, Direction, Direction]] = []
    cutter_list.append(((4, 3), (1, 0), (0, 1)))
    cutter_list.append(((4, 7), (1, 0), (0, -1)))
    cutter_list.append(((4, 9), (1, 0), (0, 1)))
    # cutter_used.append(((4, 13), (1, 0), (0, -1)))

    cutter_list.append(((6, 3), (-1, 0), (0, 1)))
    cutter_list.append(((6, 7), (-1, 0), (0, -1)))
    cutter_list.append(((6, 9), (-1, 0), (0, 1)))
    # cutter_used.append(((6, 13), (-1, 0), (0, -1)))

    cutter_list.append(((9, 3), (1, 0), (0, 1)))
    cutter_list.append(((9, 7), (1, 0), (0, -1)))
    cutter_list.append(((9, 9), (1, 0), (0, 1)))
    # cutter_used.append(((9, 13), (1, 0), (0, -1)))

    cutter_list.append(((11, 3), (-1, 0), (0, 1)))
    cutter_list.append(((11, 7), (-1, 0), (0, -1)))
    cutter_list.append(((11, 9), (-1, 0), (0, 1)))
    # cutter_used.append(((11, 13), (-1, 0), (0, -1)))
    
    router = SubProblem()
    router.route_cutters(width, height, cutter_list, starts, goals1, goals2, jump_distances, time_limit, option)