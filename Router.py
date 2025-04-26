from constants import *
from Components.Component import Component
from Components.Start import StartComponent
from Components.Border import BorderComponent
from Components.Goal import GoalComponent
from Components.Cutter import CutterComponent

from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

class Router:
    def __init__(self):
        self.components: List[Component] = []
        self.colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']

    def route_cutters(self, width, height, cutters, starts, goals1, goals2, jump_distances, timelimit, option: MIPFOCOUS_TYPE):
        # add board
        num_nets = 3
        self.initialize_board(width, height, jump_distances, num_nets)

        # compute border
        border = [(x, 0) for x in range(width)] + [(x, height-1) for x in range(width)] + [(0, y) for y in range(height)] + [(width-1, y) for y in range(height)]
        io_tiles = [node for node, _ in starts + goals1 + goals2]
        for tile in io_tiles:
            border.remove(tile)

        # add components
        self.add_borders(border)
        self.add_starts(starts, 0)
        self.add_goals(goals1, 1)
        self.add_goals(goals2, 2)
        self.add_cutters(cutters)

        # add connecting nets
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
        self.all_edges: List[Edge] = [] # for all loops
        self.step_edges: List[Edge] = [] # for cost
        self.jump_edges: List[Edge] = [] # for cost
        self.node_related_belt_edges: Dict[Node, List[Edge]] = defaultdict(list) # for is_node_used_by_belt
        
        # for adding constraints
        self.node_in_flow_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_out_flow_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_starting_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)
        self.node_related_landing_pad_edges: Dict[Node, List[Edge]] = defaultdict(list)

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

    def add_starts(self, starts: List[Tuple[Node, Direction]], net_num):
        # add edge constraints
        for node, direction in starts:
            start_object = StartComponent(node, direction, self.colors[net_num])

            # add constraints
            start_object.add_constraints(self)

            # add drawing
            self.components.append(start_object)
    
    def add_goals(self, goals: List[Tuple[Node, Direction]], net_num):
        # add edge constraints
        for node, direction in goals:
            goal_component = GoalComponent(node, direction, self.colors[net_num])

            self.components.append(goal_component)

            # add constraints
            goal_component.add_constraints(self)            

    def add_borders(self, borders: List[Node]):
        # add edge constraints
        for node in borders:
            border_component = BorderComponent(node)
        
            self.components.append(BorderComponent(node))

            border_component.add_constraints(self)

    def add_cutters(self, cutters):
        # add edge constraints
        for cutter in cutters:
            cutter_component = CutterComponent(cutter)

            self.components.append(CutterComponent(cutter))

            cutter_component.add_constraints(self)

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

        # add artists
        for artist in self.components:
            artist.draw(ax)

        for i in range(self.num_nets):
            color = self.colors[i % len(self.colors)]

            used_step_edges = [edge for edge in self.step_edges if edge in used_edge[i]]
            used_jump_edges = [edge for edge in self.jump_edges if edge in used_edge[i]]

            # plot step circule and line
            for (u, v, d) in used_step_edges:
                ux, uy = u
                ax.plot([ux + OFFSET, v[0] + OFFSET], [uy + OFFSET, v[1] + OFFSET], c='black', zorder = 1)
                ax.scatter(ux + OFFSET, uy + OFFSET, c=color, marker='o', s=50, edgecolors='black', zorder = 2)

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

                ax.plot([u2x + OFFSET, v[0] + OFFSET], [u2y + OFFSET, v[1] + OFFSET], c='black', zorder = 1)
                ax.scatter(ux + OFFSET, uy + OFFSET, c=color, marker=marker, s=80, edgecolors='black', zorder = 2)
                ax.scatter(u2x + OFFSET, u2y + OFFSET, c=color, marker=marker, s=80, edgecolors='black', zorder = 2)
        

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

        OFFSET = 0.5

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
            ax.scatter(x + OFFSET, y + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
            ax.scatter(x2 + OFFSET, y2 + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
            ax.plot([ix + OFFSET, x + OFFSET], [iy + OFFSET, y + OFFSET], c='black', zorder=0)
            ax.plot([x + OFFSET, nx + OFFSET], [y + OFFSET, ny + OFFSET], c='black', zorder=0)
            ax.plot([x2 + OFFSET, nx2 + OFFSET], [y2 + OFFSET, ny2 + OFFSET], c='black', zorder=0)

        plt.title("Shapez2: Routing using Integer Linear Programming (ILP) -- Jiahao")

        # plt.draw()
        plt.show()