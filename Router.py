# Copyright (c) 2025 Jiahao
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from constants import *
from components.Component import Component
from components.Border import BorderComponent

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
        self.ax = None
        self.terminate_requested = False
        self.flow_constrainted_node: Set[Node] = set()

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
        self.is_edge_used: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.is_node_used_by_belt: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for edge in self.all_edges:
                # is edge used (dynamic variable)
                self.is_edge_used[i][edge] = self.model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY)
                self.is_edge_used[i][edge].setAttr("BranchPriority", EDGE_PRIORITY)
        
        # edge flow value and is_edge_used indicator
        self.edge_flow_value: Dict[Edge, Var] = defaultdict(Var)
        self.node_in_flow_value_expr: Dict[Node, LinExpr] = defaultdict(LinExpr)
        self.node_out_flow_value_expr: Dict[Node, LinExpr] = defaultdict(LinExpr)
        for edge in self.all_edges:
            # edge flow value
            self.edge_flow_value[edge] = self.model.addVar(name = f"edge_flow_value_{edge}", vtype=GRB.INTEGER, lb=0, ub=FLOW_CAP)
            self.edge_flow_value[edge].setAttr("BranchPriority", FLOW_PRIORITY)
            
            # in flow and out flow values
            self.node_in_flow_value_expr[edge[1]].addTerms(1, self.edge_flow_value[edge])
            self.node_out_flow_value_expr[edge[0]].addTerms(1, self.edge_flow_value[edge])

            # if edge is used by any net, flow value must be >= 1
            for i in range(self.num_nets):
                self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[edge] >= 1)
            
            # if edge is not used by any net, flow value must be 0
            edge_used_by_any_net = self.model.addVar(name=f"edge_used_by_any_net_{edge}", vtype=GRB.BINARY)
            self.model.addGenConstrOr(edge_used_by_any_net, [self.is_edge_used[i][edge] for i in range(self.num_nets)])
            self.model.addGenConstrIndicator(edge_used_by_any_net, False, self.edge_flow_value[edge] == 0)

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
        for node in self.all_nodes:
            self.model.addConstr(self.node_in_flow_value_expr[node] <= FLOW_CAP)
            self.model.addConstr(self.node_out_flow_value_expr[node] <= FLOW_CAP)

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
        io_nodes = set()
        for component in self.components:
            io_nodes.update(component.get_nodes())  # get IO nodes from components

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

        # check if total source and sink amounts are equal
        if sum(source_amounts) != sum(sink_amounts):
            raise ValueError("Total source amounts must equal total sink amounts for a net.")

        # current net count
        i = self.current_net_count
        self.current_net_count += 1

        # register net color
        color = self.colors[i % len(self.colors)]
        for component, node in zip(source_compoents + sink_components, source_nodes + sink_nodes):
            component.register_color(node, color)

        # add flow constraints for net (within one net, flow can split and merge)
        for node in self.all_nodes:
            in_flow = self.node_in_flow_value_expr[node]
            out_flow = self.node_out_flow_value_expr[node]

            if node in source_nodes:
                source_count = source_nodes.count(node)
                self.model.addConstr(out_flow - in_flow == source_amounts[source_nodes.index(node)] * source_count)
                self.flow_constrainted_node.add(node)
            elif node in sink_nodes:
                sink_count = sink_nodes.count(node)
                self.model.addConstr(in_flow - out_flow == sink_amounts[sink_nodes.index(node)] * sink_count)
                self.flow_constrainted_node.add(node)
            else:
                # do nothing
                # then after all nets are added, set in_flow == out_flow for not used nodes
                pass

    def add_net_for_not_used_nodes(self):
        for node in self.all_nodes:
            if node not in self.flow_constrainted_node:
                in_flow = self.node_in_flow_value_expr[node]
                out_flow = self.node_out_flow_value_expr[node]

                self.model.addConstr(in_flow - out_flow == 0)

    def solve(self, timelimit, option, live_draw = False):
        # objective
        self.add_objective()

        # settings
        if timelimit != -1:
            self.model.Params.TimeLimit = timelimit
        self.model.Params.MIPFocus = option
        self.model.Params.Presolve = PRESOLVE

        # solve   
        if live_draw:  
            self.non_blocking_draw()
            self.model.optimize(self.draw_solution_callback)
        else:
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

    def draw_solution_callback(self, model, where):
        if self.terminate_requested:
            model.terminate()
            return

        if where == GRB.Callback.MIPSOL:
            used_edge = {i: [edge for edge in self.all_edges if model.cbGetSolution(self.is_edge_used[i][edge]) > 0.5] for i in range(self.num_nets)}
            self.non_blocking_draw(used_edge)

    def add_legend(self, ax):
        """Add a custom legend to the plot."""
        handle_start = Line2D([], [], marker='s', color='grey', markersize=9, markeredgecolor='black', linestyle='None')
        handle_jump_pad = Line2D([], [], marker='^', color='grey', markersize=8, markeredgecolor='black', linestyle='None')
        handle_belt = Line2D([], [], marker='o', color='grey', markersize=7, markeredgecolor='black', linestyle='None')
        legend_handles = [handle_start, handle_jump_pad, handle_belt]
        legend_labels = ['Start/Goal', 'Jump Pad', 'Belt']
        ax.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc='upper right')

    def draw_edges(self, ax, used_edge):
        # skip if no edges
        if not used_edge:
            return

        """Draw all step and jump edges."""
        for i in range(self.num_nets):
            color = self.colors[i % len(self.colors)]

            used_step_edges = [edge for edge in self.step_edges if edge in used_edge[i]]
            used_jump_edges = [edge for edge in self.jump_edges if edge in used_edge[i]]

            for (u, v, d) in used_step_edges:
                ux, uy = u
                ax.plot([ux + OFFSET, v[0] + OFFSET], [uy + OFFSET, v[1] + OFFSET], c='black', zorder=1)
                ax.scatter(ux + OFFSET, uy + OFFSET, c=color, marker='o', s=50, edgecolors='black', zorder=2)

            for (u, v, d) in used_jump_edges:
                ux, uy = u
                u2x, u2y = ux + d[0] * (max(abs(v[0] - ux), abs(v[1] - uy)) - 1), uy + d[1] * (max(abs(v[0] - ux), abs(v[1] - uy)) - 1)

                marker = {'(0, 1)': '^', '(0, -1)': 'v', '(1, 0)': '>', '(-1, 0)': '<'}.get(str(d), 'o')

                ax.plot([u2x + OFFSET, v[0] + OFFSET], [u2y + OFFSET, v[1] + OFFSET], c='black', zorder=1)
                ax.scatter(ux + OFFSET, uy + OFFSET, c=color, marker=marker, s=80, edgecolors='black', zorder=2)
                ax.scatter(u2x + OFFSET, u2y + OFFSET, c=color, marker=marker, s=80, edgecolors='black', zorder=2)

    def draw(self, used_edge = None):
        # set up axes
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_xlim(0, self.WIDTH)
        ax.set_ylim(0, self.HEIGHT)
        ax.set_xticks(range(self.WIDTH))
        ax.set_yticks(range(self.HEIGHT))
        ax.set_aspect('equal')
        ax.grid(True)

        # draw components
        for artist in self.components:
            artist.draw(ax)

        # draw edges
        self.draw_edges(ax, used_edge)

        # finalize
        plt.title("Shapez2: Routing using Mixed Integer Linear Programming (MIP)")
        self.add_legend(ax)

        # show
        plt.show()
    
    def on_close(self, event):
        self.terminate_requested = True

    def non_blocking_draw(self, used_edge = None):
        # set up axes
        if self.ax is None:
            self.fig = plt.figure(figsize=(12, 6))
            self.fig.canvas.mpl_connect('close_event', self.on_close)
            self.ax = plt.gca()
            plt.show(block=False)
        self.ax.clear()
        self.ax.set_xlim(0, self.WIDTH)
        self.ax.set_ylim(0, self.HEIGHT)
        self.ax.set_xticks(range(self.WIDTH))
        self.ax.set_yticks(range(self.HEIGHT))
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        # draw components
        for artist in self.components:
            artist.draw(self.ax)

        # draw edges
        self.draw_edges(self.ax, used_edge)

        # finalize
        plt.title("Shapez2: Routing using Mixed Integer Linear Programming (MIP)")
        self.add_legend(self.ax)

        # non blocking show
        plt.draw()
        plt.pause(0.1)