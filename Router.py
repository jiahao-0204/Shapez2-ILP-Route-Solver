from constants import *
from Components.Component import Component
from Components.Border import BorderComponent

from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List, Set
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

class Router:
    def __init__(self):
        self.components: List[Component] = []
        self.colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']
        self.current_net_count = 0

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

    def generate_and_add_borders(self) -> None:
        # collect all border nodes
        top_border = [(x, 0) for x in range(self.WIDTH)]
        bottom_border = [(x, self.HEIGHT - 1) for x in range(self.WIDTH)]
        left_border = [(0, y) for y in range(self.HEIGHT)]
        right_border = [(self.WIDTH - 1, y) for y in range(self.HEIGHT)]

        # combine and convert to set for faster operations
        border_nodes: Set[Node] = set(top_border + bottom_border + left_border + right_border)

        # collect existing IO nodes
        io_nodes = {component.node for component in self.components}

        # exclude IO nodes from borders
        border_nodes.difference_update(io_nodes)

        # create BorderComponent instances
        borders: List[BorderComponent] = [BorderComponent(node) for node in border_nodes]

        # add them
        self.add_components(borders)

    def add_components(self, components: List[Component]):
        for component in components:
            self.components.append(component)
            component.add_constraints(self)

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

    def add_sink_node_constraints(self, node:Node, input_node:Node, direction:Direction):
        for i in range(self.num_nets):
            # ------ for sink node ------

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
            
            # ------ for input node ------            
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

    def add_net(self, net_sources: List[Tuple[Component, Node, Amount]], net_sinks: List[Tuple[Component, Node, Amount]]):
        # get net sources and sinks
        source_compoents = [source[0] for source in net_sources]
        source_nodes = [source[1] for source in net_sources]
        source_amounts = [source[2] for source in net_sources]
        sink_components = [sink[0] for sink in net_sinks]
        sink_nodes = [sink[1] for sink in net_sinks]
        sink_amounts = [sink[2] for sink in net_sinks]

        # current net count
        i = self.current_net_count
        self.current_net_count += 1

        # register net color
        color = self.colors[i % len(self.colors)]
        for component in source_compoents + sink_components:
            component.register_color(color)

        # add flow constraints for net (within one net, flow can split and merge)
        for node in self.all_nodes:
            in_flow = self.node_in_flow_value_expr[i][node]
            out_flow = self.node_out_flow_value_expr[i][node]

            if node in source_nodes:
                source_count = source_nodes.count(node)
                self.model.addConstr(out_flow - in_flow == source_amounts[source_nodes.index(node)] * source_count)
            elif node in sink_nodes:
                sink_count = sink_nodes.count(node)
                self.model.addConstr(in_flow - out_flow == sink_amounts[sink_nodes.index(node)] * sink_count)
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

        # draw if have solution
        if self.model.SolCount > 0:
            used_edge = {i: [edge for edge in self.all_edges if self.is_edge_used[i][edge].X > 0.5] for i in range(self.num_nets)}
            self.draw(used_edge)

    def add_objective(self):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        self.model.setObjective(quicksum(step_cost_list + jump_cost_list))

    def draw(self, used_edge):
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

    def draw_board(self):
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