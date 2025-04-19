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

                    # skip if primary location is invalid (in in border)
                    if (x, y) not in self.all_nodes or (x, y) in self.border or (x, y) in self.port_location:
                        continue

                    # skip if secondary location is invalid
                    if (x2, y2) not in self.all_nodes or (x2, y2) in self.border or (x2, y2) in self.port_location:
                        continue

                    # skip if input location is invalid
                    if (ix, iy) not in self.all_nodes or (ix, iy) in self.border:
                        continue

                    # skip if input location is at port center
                    if (ix, iy) in self.port_center_location:
                        continue

                    # skip if output location is invalid
                    if (ox1, oy1) not in self.all_nodes or (ox2, oy2) not in self.all_nodes or (ox1, oy1) in self.border or (ox2, oy2) in self.border:
                        continue

                    # skip if output location is at corner
                    if (ox1, oy1) in self.corner or (ox2, oy2) in self.corner:
                        continue

                    # ski pif output location is at port location
                    if (ox1, oy1) in self.port_location or (ox2, oy2) in self.port_location:
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
        
        self.add_node_is_used_by_component_parts()

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
                if (nx, ny) in self.all_nodes:
                    edge = ((x, y), (nx, ny), (dx, dy))
                    self.all_edges.append(edge)
                    self.step_edges.append(edge)
                    self.node_related_step_edges[node].append(edge)
                
                for jump_distance in self.jump_distances:
                    nx, ny = x + dx * (jump_distance + 2), y + dy * (jump_distance + 2)
                    jx, jy = x + dx * (jump_distance + 1), y + dy * (jump_distance + 1)
                    pad_node = (jx, jy)
                    if (nx, ny) in self.all_nodes:
                        edge = ((x, y), (nx, ny), (dx, dy))
                        self.all_edges.append(edge)
                        self.jump_edges.append(edge)
                        self.node_related_jump_edges[node].append(edge)
                        self.node_related_jump_edges[pad_node].append(edge)

        # master problem cost
        # set objective to be number of nodes occupied by primary, secondary sources and input location
        node_used_by_component_io_bool_list = []
        for node in self.all_nodes:
            node_used_by_component_io_bool_list.append(self.node_used_by_input_location_bool[node])
            node_used_by_component_io_bool_list.append(self.node_used_by_source_bool[node])
            node_used_by_component_io_bool_list.append(self.node_used_by_secondary_source_bool[node])

        # sub problem cost
        self.sub_problem_cost = self.model.addVar(lb=0.0, name="sub_problem_cost")

        # set objective
        self.model.setObjective(quicksum(node_used_by_component_io_bool_list) + self.sub_problem_cost, GRB.MINIMIZE)

        self.add_component_count_constraint()
        self.add_component_basic_overlap_constraints()
        self.add_component_source_sink_overlap_constraints()
        self.add_component_isolation_constraints()
        self.add_component_pre_placement_constraint()

        self.model.Params.LazyConstraints = 1
        if self.timelimit != -1:
            self.model.setParam('TimeLimit', self.timelimit)
        self.model.setParam('MIPFocus', self.option)
        self.model.setParam('Presolve', 2)
        self.model.setParam('Heuristics', 0.5)

        self.model.optimize(self.benders_callback)

        # self.model.optimize()

        is_component_used = {component: self.is_component_used[component].X for component in self.all_components}
        self.draw_components(is_component_used)

        # self.model.computeIIS()
        # if self.model.status == GRB.INFEASIBLE:
        #     for c in self.model.getConstrs():
        #         if c.IISConstr:
        #             print(f"Constraint {c.constrName} is in the IIS")

    def benders_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            # get master solution
            is_component_used = {component: model.cbGetSolution(self.is_component_used[component]) for component in self.all_components}

            # solve subproblem
            feasible, cost = self.solve_subproblem(is_component_used)

            # cut the component used if combination not feasible
            if not feasible:
                used_components = [c for c in self.all_components if is_component_used[c] > 0.5]
                expr = quicksum(self.is_component_used[component] for component in used_components)
                model.cbLazy(expr <= len(used_components)-1)
                return
            
            # here the solution is feasible
            model.cbLazy(self.sub_problem_cost >= cost)

            # # plot location of components
            # components_used = [component for component, value in is_component_used.items() if value > 0.5]
            # print("Components used: ", len(components_used))
            # print("whose node is: ", [component[0] for component in components_used])

            # # draw components
            # self.draw_components(is_component_used)

    def solve_subproblem(self, is_component_used):
        sub_model = Model("subproblem")
        sub_model.Params.OutputFlag = 0  # silent
        if self.timelimit != -1:
            sub_model.Params.TimeLimit = self.timelimit
        sub_model.Params.MIPFocus = self.option
        sub_model.Params.Presolve = 2

        sub_problem_data = {
            "is_edge_used": {},
            "is_node_used_by_step_edge": {},
            "edge_flow_value": {},
        }
        
        # is edge used
        for i in range(self.num_nets):
            sub_problem_data["is_edge_used"][i] = {edge: sub_model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY) for edge in self.all_edges}
            # set priority
            for edge in self.all_edges:
                sub_problem_data["is_edge_used"][i][edge].setAttr("BranchPriority", self.edge_priority)
        
        self.add_variable_is_node_used_by_step_edges(sub_model, sub_problem_data)

        # Objective function
        self.add_objective(sub_model, sub_problem_data)

        # Constraints
        self.add_constraints(sub_model, sub_problem_data, is_component_used)

        # Solve
        self.solve(sub_model)

        # get solution
        if sub_model.Status == GRB.INFEASIBLE:
            return False, None
        else:
            return True, sub_model.ObjVal

        # # Plot
        # self.plot()

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
        used_components = [c for c in self.all_components if is_component_used[c] > 0.5]
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

    def add_variable_is_node_used_by_step_edges(self, sub_model, sub_problem_data):
        sub_problem_data["is_node_used_by_step_edge"] = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for node in self.all_nodes:
                node_step_edges_bool_list = [sub_problem_data["is_edge_used"][i][edge] for edge in self.node_related_step_edges[node]]
                sub_problem_data["is_node_used_by_step_edge"][i][node] = sub_model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)
                sub_model.addGenConstrOr(sub_problem_data["is_node_used_by_step_edge"][i][node], node_step_edges_bool_list)

    def add_objective(self, sub_model, sub_problem_data):
        step_cost_list = []
        jump_cost_list = []

        for i in range(self.num_nets):
            step_cost_list_i = [sub_problem_data["is_edge_used"][i][edge] * STEP_COST for edge in self.step_edges]
            jump_cost_list_i = [sub_problem_data["is_edge_used"][i][edge] * JUMP_COST for edge in self.jump_edges]
            step_cost_list += step_cost_list_i
            jump_cost_list += jump_cost_list_i
            
        sub_model.setObjective(quicksum(step_cost_list + jump_cost_list))

    def add_constraints(self, sub_model, sub_problem_data, is_component_used):
        self.add_flow_constraints_source_to_components(0, sub_model, sub_problem_data, is_component_used)
        self.add_flow_constraints_component_to_goal(1, sub_model, sub_problem_data, is_component_used)
        self.add_flow_constraints_secondary_component_to_goal(2, sub_model, sub_problem_data, is_component_used)

        for i in range(self.num_nets):
            # self.add_no_step_jump_overlap_constraints(i)
            self.add_directional_constraints_w_component(i, sub_model, sub_problem_data, is_component_used)

        self.add_things_overlap_constraints(sub_model, sub_problem_data, is_component_used)

    def add_flow_constraints_source_to_components(self, i, sub_model, sub_problem_data, is_component_used):
        sub_problem_data["edge_flow_value"] = {}
        sub_problem_data["edge_flow_value"][i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            sub_problem_data["edge_flow_value"][i][edge].setAttr("BranchPriority", self.flow_priority)
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], True, sub_problem_data["edge_flow_value"][i][edge] >= 1)
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], False, sub_problem_data["edge_flow_value"][i][edge] == 0)

        sources = self.net_sources[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_components[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_sink = any(node_component_used_bool_list)

            in_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sources:
                sub_model.addConstr(in_flow == 0)
                sub_model.addConstr(out_flow == self.start_amount)
            else:
                if node_is_component_sink:
                    sub_model.addConstr(in_flow - out_flow == self.component_sink_amount)
                else:
                    sub_model.addConstr(in_flow - out_flow == 0)
    
    def add_flow_constraints_component_to_goal(self, i, sub_model, sub_problem_data, is_component_used):
        sub_problem_data["edge_flow_value"] = {}
        sub_problem_data["edge_flow_value"][i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            sub_problem_data["edge_flow_value"][i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], True, sub_problem_data["edge_flow_value"][i][edge] >= 1)
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], False, sub_problem_data["edge_flow_value"][i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_sources[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_source = any(node_component_used_bool_list)

            in_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                sub_model.addConstr(out_flow == 0)
                sub_model.addConstr(in_flow == self.goal_amount)
            else:
                if node_is_component_source:
                    sub_model.addConstr(out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))
                else:
                    sub_model.addConstr(out_flow - in_flow == 0)

    def add_flow_constraints_secondary_component_to_goal(self, i, sub_model, sub_problem_data, is_component_used):
        sub_problem_data["edge_flow_value"] = {}
        sub_problem_data["edge_flow_value"][i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            sub_problem_data["edge_flow_value"][i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], True, sub_problem_data["edge_flow_value"][i][edge] >= 1)
            sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], False, sub_problem_data["edge_flow_value"][i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_secondary_sources[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_source = any(node_component_used_bool_list)

            in_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(sub_problem_data["edge_flow_value"][i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                sub_model.addConstr(out_flow == 0)
                sub_model.addConstr(in_flow == self.goal_amount)
            else:
                if node_is_component_source:
                    sub_model.addConstr(out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))
                else:
                    sub_model.addConstr(out_flow - in_flow == 0)
                  
    def add_directional_constraints_w_component(self, i, sub_model, sub_problem_data, is_component_used):
        # no jump edge at start
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                sub_model.addConstr(sub_problem_data["is_edge_used"][i][jump_edge] == 0)
        
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
                        # sub_model.addGenConstrIndicator(is_component_used[component], True, sub_problem_data["is_edge_used"][i][jump_edge] == 0)
                        if is_component_used[component] > 0.5:
                            sub_model.addConstr(sub_problem_data["is_edge_used"][i][jump_edge] == 0)
            
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
                        # sub_model.addGenConstrIndicator(is_component_used[component], True, sub_problem_data["is_edge_used"][i][jump_edge] == 0)
                        if is_component_used[component] > 0.5:
                            sub_model.addConstr(sub_problem_data["is_edge_used"][i][jump_edge] == 0)

        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for edge in self.all_edges:
            u, v, direction = edge

            # if the edge is used, then the end node must not have starting jump edge at different direction, and must not have any landing jump edge
            for jump_edge in self.node_related_jump_edges[v]:
                u2, v2, jump_direction = jump_edge
                if u2 == v and direction == jump_direction: # starting jump edge
                    continue
                else:
                    # sub_model.addGenConstrIndicator(sub_problem_data["is_edge_used"][i][edge], True, sub_problem_data["is_edge_used"][i][jump_edge] == 0)
                    sub_model.addConstr(sub_problem_data["is_edge_used"][i][edge] + sub_problem_data["is_edge_used"][i][jump_edge] <= 1) # only one can be true

    def add_things_overlap_constraints(self, sub_model, sub_problem_data, is_component_used):
        # between belts / pads / components
        for node in self.all_nodes:
            list_of_things_using_node = []
            occupied = False
            # for i in range(self.num_nets):
            #     list_of_things_using_node.append(self.is_node_used_by_net[i][node])
            for i in range(self.num_nets):
                list_of_things_using_node.append(sub_problem_data["is_node_used_by_step_edge"][i][node])
                list_of_things_using_node += [sub_problem_data["is_edge_used"][i][edge] for edge in self.node_related_jump_edges[node]]
            for component in self.node_related_components[node]:
                if is_component_used[component] > 0.5:
                    occupied = True
            for component in self.node_related_secondary_components[node]:
                if is_component_used[component] > 0.5:
                    occupied = True
            
            if occupied:
                # no things can occupy this
                sub_model.addConstr(quicksum(list_of_things_using_node) == 0)
            else:
                # constraint: at most one thing can use a node
                sub_model.addConstr(quicksum(list_of_things_using_node) <= 1)

    def add_component_count_constraint(self):
        # add component count constraint
        component_used_bool_list = [self.is_component_used[component] for component in self.all_components]
        self.model.addConstr(quicksum(component_used_bool_list) == self.component_count, name="component_count_constraint")

    def add_component_pre_placement_constraint(self):
        preplacement_list = []

        # preplacement_list.append(((6, 8), (0, 1), (1, 0)))
        # preplacement_list.append(((7, 7), (1, 0), (0, -1)))
        # # preplacement_list.append(((6, 6), (0, -1), (-1, 0)))
        # preplacement_list.append(((5, 7), (-1, 0), (0, 1)))

        # preplacement_list.append(((3, 2), (0, 1), (-1, 0)))
        # preplacement_list.append(((6, 3), (0, 1), (1, 0)))
        # preplacement_list.append(((10, 3), (0, 1), (-1, 0)))
        # preplacement_list.append(((12, 3), (0, 1), (1, 0)))

        # preplacement_list.append(((3, 4), (0, -1), (-1, 0)))
        # preplacement_list.append(((6, 5), (0, -1), (1, 0)))
        # preplacement_list.append(((10, 5), (0, -1), (-1, 0)))
        # preplacement_list.append(((12, 5), (0, -1), (1, 0)))

        # # preplacement_list.append(((5, 10), (0, -1), (-1, 0)))
        # # preplacement_list.append(((5, 11), (-1, 0), (0, 1))) # this next
        # # preplacement_list.append(((5, 12), (0, 1), (1, 0))) # this next 2
        # preplacement_list.append(((6, 8), (0, 1), (1, 0)))
        # preplacement_list.append(((10, 8), (0, 1), (-1, 0)))
        # preplacement_list.append(((12, 8), (0, 1), (1, 0)))

        # # preplacement_list.append(((3, 12), (0, 1), (-1, 0))) # this next 2
        # preplacement_list.append(((6, 10), (0, -1), (1, 0)))
        # preplacement_list.append(((10, 10), (0, -1), (-1, 0)))
        # preplacement_list.append(((12, 10), (0, -1), (1, 0)))

        for component in preplacement_list:
            # add constraint
            self.model.addConstr(self.is_component_used[component] == 1, name=f"component_pre_placement_{component}")

        # for component in self.all_components:
        #     if component in preplacement_list:
        #         is_component_used[component].Start = 1

    def add_component_basic_overlap_constraints(self):
        # between components
        for node in self.all_nodes:
            list_of_things_using_node = []
            for component in self.node_related_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            for component in self.node_related_secondary_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            # constraint: at most one thing can use a node
            self.model.addConstr(quicksum(list_of_things_using_node) <= 1, name=f"component_overlap_{node}")

    def add_node_is_used_by_component_parts(self):
        self.node_used_by_primary_component_bool = {node: self.model.addVar(name=f"node_used_by_primary_component_bool_{node}", vtype=GRB.BINARY) for node in self.all_nodes}
        self.node_used_by_secondary_component_bool = {node: self.model.addVar(name=f"node_used_by_secondary_component_bool_{node}", vtype=GRB.BINARY) for node in self.all_nodes}
        self.node_used_by_source_bool = {node: self.model.addVar(name=f"node_used_by_source_bool_{node}", vtype=GRB.BINARY) for node in self.all_nodes}
        self.node_used_by_secondary_source_bool = {node: self.model.addVar(name=f"node_used_by_secondary_source_bool_{node}", vtype=GRB.BINARY) for node in self.all_nodes}
        self.node_used_by_input_location_bool = {node: self.model.addVar(name=f"node_used_by_input_location_bool_{node}", vtype=GRB.BINARY) for node in self.all_nodes}

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

            # OR 
            self.model.addGenConstrOr(self.node_used_by_primary_component_bool[node], node_used_by_primary_component_bool_list)
            self.model.addGenConstrOr(self.node_used_by_secondary_component_bool[node], node_used_by_secondary_component_bool_list)
            self.model.addGenConstrOr(self.node_used_by_source_bool[node], node_used_by_source_bool_list)
            self.model.addGenConstrOr(self.node_used_by_secondary_source_bool[node], node_used_by_secondary_source_bool_list)
            self.model.addGenConstrOr(self.node_used_by_input_location_bool[node], node_used_by_input_location_bool_list)


    def add_component_source_sink_overlap_constraints(self):
        for node in self.all_nodes:
            # only one can be true
            self.model.addConstr(quicksum([
                self.node_used_by_primary_component_bool[node],
                self.node_used_by_secondary_component_bool[node],
                self.node_used_by_source_bool[node],
                self.node_used_by_secondary_source_bool[node],
                self.node_used_by_input_location_bool[node]
            ]) <= 1, name=f"component_source_sink_overlap_{node}")

    def add_component_isolation_constraints(self):
        # These three dicts map node → list of components that place that role there
        role_list = [
            self.node_used_by_primary_component_bool,
            self.node_used_by_secondary_component_bool,
            self.node_used_by_source_bool,
            self.node_used_by_secondary_source_bool,
            self.node_used_by_input_location_bool
        ]

        for i, role in enumerate(role_list):
            # skip if is primary and secondary component
            if i == 0 or i == 1:
                continue

            # for each node
            for node in self.all_nodes:
                
                # neighboring nodes
                neighbor_nodes = [(node[0] + dx, node[1] + dy) for dx, dy in DIRECTIONS if (node[0] + dx, node[1] + dy) in self.all_nodes]

                # neighboring nodes used by other role
                other_role_neighbors_bool_list = [] 
                for other_role in role_list:
                    if other_role is role:
                        continue
                    for neighbor_node in neighbor_nodes:
                        other_role_neighbors_bool_list.append(other_role[neighbor_node])

                # if this node is used by this role, then the neighboring nodes must not all be used by other role
                self.model.addGenConstrIndicator(role[node], True, quicksum(other_role_neighbors_bool_list) <= len(neighbor_nodes) - 1, name=f"component_isolation_{node}_{i}")

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

    def plot(self, sub_problem_data, is_component_used):
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

            used_step_edges = [e for e in self.step_edges if sub_problem_data["is_edge_used"][i][e].X == 1]
            used_jump_edges = [e for e in self.jump_edges if sub_problem_data["is_edge_used"][i][e].X == 1]

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
        used_components = [c for c in self.all_components if is_component_used[c].X == 1]
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
    router = DirectionalJumpRouter(width=16, height=16, nets=nets, jump_distances= [1, 2, 3, 4], timelimit = -1, symmetry = False, option = 3)
    # option 0: balanced
    # option 1: feasibility
    # option 2: bound
    # option 3: incumbent improvement


    # 169 cost without variable length launcher for 8 to 32 routing
    # # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/564a50812cc64b7f96e235e5bd1d7f8e-pulp.sol

    # 40 cost for cutter in 1x1
    # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/cab48f13fec049babcf0b53d7f8d3c99-pulp.sol