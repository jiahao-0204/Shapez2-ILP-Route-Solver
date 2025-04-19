from gurobipy import Model, GRB, quicksum, Var

import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, Tuple, List
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
JUMP_COST = 2
STEP_COST = 1

Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Amount = int # amount of flow

Edge = Tuple[Node, Node, Direction] # start, end, direciton
Component = Tuple[Node, Direction] # location, direction

class DirectionalJumpRouter:
    def __init__(self, width, height, nets, jump_distances: List[int] = [4], timelimit: int = 60, symmetry: bool = False, option: int = 0):

        # allow multiple start

        # Input parameters
        self.WIDTH = width
        self.HEIGHT = height
        self.jump_distances = jump_distances
        self.timelimit = timelimit
        self.symmetry = symmetry
        self.option = option

        self.num_nets = 3 # start to component, component to goal
        self.net_sources: Dict[int, List[Node]] = {}
        self.net_sinks: Dict[int, List[Node]] = {}

        sources, sinks1, sinks2 = nets[0]
        
        self.net_sources[0] = sources
        self.net_sinks[0] = []

        self.net_sources[1] = []
        self.net_sinks[1] = sinks1

        self.net_sources[2] = []
        self.net_sinks[2] = sinks2

        self.flow_cap = 4
        self.start_amount = 4
        self.goal_amount = 4
        self.component_source_amount = 1
        self.component_sink_amount = 1
        # self.component_count = 12
        self.component_count = len(self.net_sources[0]) * (4/self.component_source_amount)
        self.component_priority = 100
        self.edge_priority = 50
        self.flow_priority = 25

        # Optimization
        self.model = Model("DirectionalJumpRouter")

        # blocked tile is the border of the map
        self.blocked_tiles = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
        remove_from_blocked_tiles = [(6, 0), (7, 0), (8, 0), (9, 0)]
        remove_from_blocked_tiles += [(6, self.HEIGHT-1), (7, self.HEIGHT-1), (8, self.HEIGHT-1), (9, self.HEIGHT-1)]
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

        # all possible location and orientation to place the components
        self.all_components: List[Component] = []
        self.node_related_components: Dict[Node, List[Component]] = defaultdict(list) # to record occupancy
        self.node_related_secondary_components: Dict[Node, List[Component]] = defaultdict(list) # to record occupancy of secondary component
        self.node_related_component_sources: Dict[Node, List[Component]] = defaultdict(list) # to record source
        self.node_related_component_secondary_sources: Dict[Node, List[Component]] = defaultdict(list) # to record secondary source
        self.node_related_component_sinks: Dict[Node, List[Component]] = defaultdict(list) # to record sink
        self.node_related_component_input_location: Dict[Node, List[Component]] = defaultdict(list) # to record input location
        for node in self.all_nodes:
            x, y = node
            for dx, dy in DIRECTIONS:
                # compute secondary direction: direction but exclude the opposite direction and the same direction
                secondary_direction = DIRECTIONS.copy()
                secondary_direction.remove((dx, dy))
                secondary_direction.remove((-dx, -dy))
                
                for secondary_dx, secondary_dy in secondary_direction:
                    component = ((x, y), (dx, dy), (secondary_dx, secondary_dy))

                    # secondary location
                    x2 = x + secondary_dx
                    y2 = y + secondary_dy

                    # input location (sink)
                    ix = x - dx
                    iy = y - dy

                    # output location (source)
                    ox1 = x + dx
                    oy1 = y + dy
                    ox2 = x + secondary_dx + dx
                    oy2 = y + secondary_dy + dy

                    # skip if secondary location is invalid
                    if (x2, y2) not in self.all_nodes:
                        continue

                    # skip if input location is invalid
                    if (ix, iy) not in self.all_nodes:
                        continue

                    # skip if output location is invalid
                    if (ox1, oy1) not in self.all_nodes or (ox2, oy2) not in self.all_nodes:
                        continue

                    self.all_components.append(component)
                    self.node_related_components[(x, y)].append(component)
                    self.node_related_secondary_components[(x2, y2)].append(component)
                    self.node_related_component_sources[(ox1, oy1)].append(component)
                    self.node_related_component_secondary_sources[(ox2, oy2)].append(component)
                    self.node_related_component_sinks[(x, y)].append(component)
                    self.node_related_component_input_location[(ix, iy)].append(component)

        # is component used
        self.is_component_used: Dict[Component, Var] = {}
        self.is_component_used = {component: self.model.addVar(name=f"component_{component}", vtype=GRB.BINARY) for component in self.all_components}
        # set priority
        for comonent in self.all_components:
            self.is_component_used[comonent].setAttr("BranchPriority", self.component_priority)
        
        # self.all_edges: List[Edge] = []
        # self.step_edges: List[Edge] = []
        # self.jump_edges: List[Edge] = []
        # self.node_related_step_edges: Dict[Node, List[Edge]] = defaultdict(list)
        # self.node_related_jump_edges: Dict[Node, List[Edge]] = defaultdict(list)
        # for node in self.all_nodes:
        #     x, y = node
        #     for dx, dy in DIRECTIONS:

        #         # Step edge
        #         nx, ny = x + dx, y + dy
        #         if (nx, ny) in self.all_nodes:
        #             edge = ((x, y), (nx, ny), (dx, dy))
        #             self.all_edges.append(edge)
        #             self.step_edges.append(edge)
        #             self.node_related_step_edges[node].append(edge)
                
        #         for jump_distance in self.jump_distances:
        #             nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
        #             jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
        #             pad_node = (jx, jy)
        #             if (nx, ny) in self.all_nodes:
        #                 edge = ((x, y), (nx, ny), (dx, dy))
        #                 self.all_edges.append(edge)
        #                 self.jump_edges.append(edge)
        #                 self.node_related_jump_edges[node].append(edge)
        #                 self.node_related_jump_edges[pad_node].append(edge)

        
        
        # # is edge used
        # self.is_edge_used: Dict[int, Dict[Edge, Var]] = {}
        # for i in range(self.num_nets):
        #     self.is_edge_used[i] = {edge: self.model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY) for edge in self.all_edges}
        #     # set priority
        #     for edge in self.all_edges:
        #         self.is_edge_used[i][edge].setAttr("BranchPriority", self.edge_priority)
        
        # self.add_variable_is_node_used_by_step_edges()

        self.sub_problem_cost = self.model.addVar(lb=0.0, name="sub_problem_cost")
        self.model.setObjective(self.sub_problem_cost, GRB.MINIMIZE)
        self.add_component_count_constraint()
        self.add_component_basic_overlap_constraints()
        self.add_component_source_sink_overlap_constraints()
        self.add_component_pre_placement_constraint()

        self.model.Params.LazyConstraints = 1
        if self.timelimit != -1:
            self.model.setParam('TimeLimit', self.timelimit)
        self.model.setParam('MIPFocus', self.option)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.5)

        # Optimize with Benders callback
        self.model.optimize(self.benders_callback)

        # # Objective function
        # self.add_objective()

        # # Constraints
        # self.add_constraints()

        # # Solve
        # self.solve()

        # # Plot
        # self.plot()

    def benders_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            # get solution
            is_component_used = {component: model.cbGetSolution(self.is_component_used[component]) for component in self.all_components}

            # draw components
            self.draw_components(is_component_used)

            # cut the component used
            used_components = [c for c in self.all_components if is_component_used[c] > 0.5]
            expr = quicksum(self.is_component_used[component] for component in used_components)
            model.cbLazy(expr <= len(used_components)-1)


    def draw_components(self, is_component_used):
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

        # draw components
        used_components = [c for c in self.all_components if is_component_used[c] == 1]
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

        # show
        plt.show()

    def add_variable_is_node_used_by_step_edges(self):
        self.is_node_used_by_step_edge: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for node in self.all_nodes:
                node_step_edges_bool_list = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
                self.is_node_used_by_step_edge[i][node] = self.model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)
                self.model.addGenConstrOr(self.is_node_used_by_step_edge[i][node], node_step_edges_bool_list)

    def add_objective(self):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [self.is_edge_used[i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [self.is_edge_used[i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        self.model.setObjective(quicksum(step_cost_list + jump_cost_list))

    def add_constraints(self):
        self.add_flow_constraints_source_to_components(0)
        self.add_flow_constraints_component_to_goal(1)
        self.add_flow_constraints_secondary_component_to_goal(2)

        for i in range(self.num_nets):
            # self.add_no_step_jump_overlap_constraints(i)
            self.add_directional_constraints_w_component(i)

        self.add_things_overlap_constraints()

        if self.symmetry:
            self.add_symmetry_constraints()

    def add_flow_constraints_source_to_components(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sources = self.net_sources[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_components[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_sink = self.model.addVar(name = f"node_is_component_sink_{i}_{node}", vtype=GRB.BINARY)
            self.model.addGenConstrOr(node_is_component_sink, node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= self.flow_cap)
            self.model.addConstr(out_flow <= self.flow_cap)

            if node in sources:
                self.model.addConstr(in_flow == 0)
                self.model.addConstr(out_flow == self.start_amount)
            else:
                self.model.addGenConstrIndicator(node_is_component_sink, True, in_flow - out_flow == self.component_sink_amount)
                self.model.addGenConstrIndicator(node_is_component_sink, False, in_flow - out_flow == 0)
    
    def add_flow_constraints_component_to_goal(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_sources[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_source = self.model.addVar(name = f"node_is_component_source_{i}_{node}", vtype=GRB.BINARY)
            self.model.addGenConstrOr(node_is_component_source, node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= self.flow_cap)
            self.model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                self.model.addConstr(out_flow == 0)
                self.model.addConstr(in_flow == self.goal_amount)
            else:
                self.model.addGenConstrIndicator(node_is_component_source, True, out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))                
                self.model.addGenConstrIndicator(node_is_component_source, False, out_flow - in_flow == 0)

    def add_flow_constraints_secondary_component_to_goal(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            self.model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_secondary_sources[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_source = self.model.addVar(name = f"node_is_component_source_{i}_{node}", vtype=GRB.BINARY)
            self.model.addGenConstrOr(node_is_component_source, node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= self.flow_cap)
            self.model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                self.model.addConstr(out_flow == 0)
                self.model.addConstr(in_flow == self.goal_amount)
            else:
                self.model.addGenConstrIndicator(node_is_component_source, True, out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))
                self.model.addGenConstrIndicator(node_is_component_source, False, out_flow - in_flow == 0)
                  
    def add_directional_constraints_w_component(self, i):
        # no jump edge at start
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                self.model.addConstr(self.is_edge_used[i][jump_edge] == 0)
        
        # component direction constraint
        # for each node
        for node in self.all_nodes:
            # for each possible component source that can be placed at this node
            for component in self.node_related_component_sources[node] + self.node_related_component_secondary_sources[node]:
                _, component_direction, _ = component

                # if this component is active, jumps in different direction are not allowed
                for jump_edge in self.node_related_jump_edges[node]:
                    u, v, direction = jump_edge
                    if u == node and direction == component_direction:
                        continue
                    else:
                        # self.model.addGenConstrIndicator(self.is_component_used[component], True, self.is_edge_used[i][jump_edge] == 0)
                        self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)
            
            # for each possible component sink that can be placed at this node
            for component in self.node_related_component_sinks[node]:
                _, component_direction, _ = component
                related_jump_edge = [edge for edge in self.all_edges if edge[1] == node]

                # if this component is active, jumps in different direction are not allowed
                for jump_edge in related_jump_edge:
                    u, v, direction = jump_edge
                    if v == node and direction == component_direction:
                        continue
                    else:
                        # self.model.addGenConstrIndicator(self.is_component_used[component], True, self.is_edge_used[i][jump_edge] == 0)
                        self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)

        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for edge in self.all_edges:
            u, v, direction = edge

            # if the edge is used, then the end node must not have starting jump edge at different direction, and must not have any landing jump edge
            for jump_edge in self.node_related_jump_edges[v]:
                u2, v2, jump_direction = jump_edge
                if u2 == v and direction == jump_direction: # starting jump edge
                    continue
                else:
                    # self.model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.is_edge_used[i][jump_edge] == 0)
                    self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

    def add_things_overlap_constraints(self):
        # between belts / pads / components
        for node in self.all_nodes:
            list_of_things_using_node = []
            # for i in range(self.num_nets):
            #     list_of_things_using_node.append(self.is_node_used_by_net[i][node])
            for i in range(self.num_nets):
                list_of_things_using_node.append(self.is_node_used_by_step_edge[i][node])
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]
            for component in self.node_related_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            for component in self.node_related_secondary_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            
            # constraint: at most one thing can use a node
            self.model.addConstr(quicksum(list_of_things_using_node) <= 1)

    def add_component_count_constraint(self):
        # add component count constraint
        component_used_bool_list = [self.is_component_used[component] for component in self.all_components]
        self.model.addConstr(quicksum(component_used_bool_list) == self.component_count)

    def add_component_pre_placement_constraint(self):
        preplacement_list = []

        # preplacement_list.append(((3, 2), (0, 1), (-1, 0)))
        preplacement_list.append(((6, 3), (0, 1), (1, 0)))
        preplacement_list.append(((10, 3), (0, 1), (-1, 0)))
        preplacement_list.append(((12, 3), (0, 1), (1, 0)))

        # preplacement_list.append(((3, 4), (0, -1), (-1, 0)))
        preplacement_list.append(((6, 5), (0, -1), (1, 0)))
        preplacement_list.append(((10, 5), (0, -1), (-1, 0)))
        preplacement_list.append(((12, 5), (0, -1), (1, 0)))

        # preplacement_list.append(((5, 10), (0, -1), (-1, 0)))
        # preplacement_list.append(((5, 11), (-1, 0), (0, 1))) # this next
        # preplacement_list.append(((5, 12), (0, 1), (1, 0))) # this next 2
        preplacement_list.append(((6, 8), (0, 1), (1, 0)))
        preplacement_list.append(((10, 8), (0, 1), (-1, 0)))
        preplacement_list.append(((12, 8), (0, 1), (1, 0)))

        # preplacement_list.append(((3, 12), (0, 1), (-1, 0))) # this next 2
        preplacement_list.append(((6, 10), (0, -1), (1, 0)))
        preplacement_list.append(((10, 10), (0, -1), (-1, 0)))
        preplacement_list.append(((12, 10), (0, -1), (1, 0)))

        for component in preplacement_list:
            # add constraint
            self.model.addConstr(self.is_component_used[component] == 1)

        # for component in self.all_components:
        #     if component in preplacement_list:
        #         self.is_component_used[component].Start = 1

    def add_component_basic_overlap_constraints(self):
        # between components
        for node in self.all_nodes:
            list_of_things_using_node = []
            for component in self.node_related_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            for component in self.node_related_secondary_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            # constraint: at most one thing can use a node
            self.model.addConstr(quicksum(list_of_things_using_node) <= 1)

    def add_component_source_sink_overlap_constraints(self):
        for node in self.all_nodes:
            # node related component parts
            node_related_primary_component = self.node_related_components[node]
            node_related_secondary_component = self.node_related_secondary_components[node]
            node_related_sources = self.node_related_component_sources[node]
            node_related_secondary_sources = self.node_related_component_secondary_sources[node]
            node_related_input_location = self.node_related_component_input_location[node]

            # node occupied by componet parts bool list
            node_used_by_primary_component_bool_list = [self.is_component_used[component] for component in node_related_primary_component]
            node_used_by_secondary_component_bool_list = [self.is_component_used[component] for component in node_related_secondary_component]
            node_used_by_source_bool_list = [self.is_component_used[component] for component in node_related_sources]
            node_used_by_secondary_source_bool_list = [self.is_component_used[component] for component in node_related_secondary_sources]
            node_used_by_input_location_bool_list = [self.is_component_used[component] for component in node_related_input_location]

            # node occupied by componet parts bool var
            node_used_by_primary_component_bool = self.model.addVar(name = f"node_used_by_primary_component_bool_{node}", vtype=GRB.BINARY)
            node_used_by_secondary_component_bool = self.model.addVar(name = f"node_used_by_secondary_component_bool_{node}", vtype=GRB.BINARY)
            node_used_by_source_bool = self.model.addVar(name = f"node_used_by_source_bool_{node}", vtype=GRB.BINARY)
            node_used_by_secondary_source_bool = self.model.addVar(name = f"node_used_by_secondary_source_bool_{node}", vtype=GRB.BINARY)
            node_used_by_input_location_bool = self.model.addVar(name = f"node_used_by_input_location_bool_{node}", vtype=GRB.BINARY)

            # OR 
            self.model.addGenConstrOr(node_used_by_primary_component_bool, node_used_by_primary_component_bool_list)
            self.model.addGenConstrOr(node_used_by_secondary_component_bool, node_used_by_secondary_component_bool_list)
            self.model.addGenConstrOr(node_used_by_source_bool, node_used_by_source_bool_list)
            self.model.addGenConstrOr(node_used_by_secondary_source_bool, node_used_by_secondary_source_bool_list)
            self.model.addGenConstrOr(node_used_by_input_location_bool, node_used_by_input_location_bool_list)

            # only one can be true
            self.model.addConstr(quicksum([node_used_by_primary_component_bool, 
                                           node_used_by_secondary_component_bool, 
                                           node_used_by_source_bool, 
                                           node_used_by_secondary_source_bool, 
                                           node_used_by_input_location_bool]) <= 1)

    def solve(self):
        if self.timelimit != -1:
            self.model.setParam('TimeLimit', self.timelimit)
        self.model.setParam('MIPFocus', self.option)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.5)
        self.model.update()
        self.model.optimize()
        
        # # Copy and apply feasibility relaxation
        # relaxed_model = self.model.copy()
        # relaxed_model.feasRelaxS(0, False, False, True)
        # relaxed_model.optimize()

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

        # draw blocked tiles
        for (x, y) in self.blocked_tiles:
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='none', hatch='////'))
            # ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', edgecolor='black', linewidth=2))
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', linewidth=2))


        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'gray', 'olive']    
        for i in range(self.num_nets):
            color = colors[i % len(colors)]

            used_step_edges = [e for e in self.step_edges if self.is_edge_used[i][e].X == 1]
            used_jump_edges = [e for e in self.jump_edges if self.is_edge_used[i][e].X == 1]

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
        used_components = [c for c in self.all_components if self.is_component_used[c].X == 1]
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


# Example usage
if __name__ == "__main__":
    nets = [        
        # try aws again, this time use this https://support.gurobi.com/hc/en-us/articles/13232844297489-How-do-I-set-up-a-Web-License-Service-WLS-license

        # regular output but up and left reversed
        # ([(6, 0)], 
        #  [(0, 6)],
        #  [(6, 15)]),

        # ([(6, 0), (7, 0)], 
        # [(0, 6), (0, 7)],
        # [(6, 15), (7, 15)]),

        # ([(6, 0), (8, 0)], 
        #  [(0, 6), (0, 8)],
        #  [(6, 15), (8, 15)]),

        # ([(6, 0), (7, 0), (8, 0)], 
        #  [(0, 6), (0, 7), (0, 8)],
        #  [(6, 15), (7, 15), (8, 15)]),

        # ([(7, 0), (8, 0), (9, 0)], 
        #  [(0, 7), (0, 8), (0, 9)],
        #  [(7, 15), (8, 15), (9, 15)]),

        ([(6, 0), (7, 0), (8, 0), (9, 0)], 
         [(0, 6), (0, 7), (0, 8), (0, 9)],
         [(6, 15), (7, 15), (8, 15), (9, 15)]),

        ]
    router = DirectionalJumpRouter(width=16, height=16, nets=nets, jump_distances= [1, 2, 3, 4], timelimit = -1, symmetry = False, option = 1)
    # option 0: balanced
    # option 1: feasibility
    # option 2: bound
    # option 3: incumbent improvement


    # 169 cost without variable length launcher for 8 to 32 routing
    # # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/564a50812cc64b7f96e235e5bd1d7f8e-pulp.sol

    # 40 cost for cutter in 1x1
    # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/cab48f13fec049babcf0b53d7f8d3c99-pulp.sol