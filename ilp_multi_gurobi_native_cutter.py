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

Source = Tuple[Node, Direction, Amount] # location, direction, amount
Sink = Tuple[Node, Direction, Amount] # location, direction, amount

class DirectionalJumpRouter:
    def __init__(self, width, height, nets, jump_distances: List[int] = [4], timelimit: int = 60, symmetry: bool = False, use_option: bool = False):

        # allow multiple start

        # Input parameters
        self.WIDTH = width
        self.HEIGHT = height

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

        self.component_source_amount = 1
        self.component_sink_amount = 1

        # Optimization
        self.model = Model("DirectionalJumpRouter")

        # blocked tile is the border of the map
        self.blocked_tiles = [(x, 0) for x in range(self.WIDTH)] + [(x, self.HEIGHT-1) for x in range(self.WIDTH)] + [(0, y) for y in range(self.HEIGHT)] + [(self.WIDTH-1, y) for y in range(self.HEIGHT)]
        remove_from_blocked_tiles = [(6, 0), (7, 0), (8, 0), (9, 0)]
        remove_from_blocked_tiles += [(6, 15), (7, 15), (8, 15), (9, 15)]
        remove_from_blocked_tiles += [(0, 6), (0, 7), (0, 8), (0, 9)]
        for tile in remove_from_blocked_tiles:
            self.blocked_tiles.remove(tile)

        # all nodes
        self.all_nodes: List[Node] = []
        for x in range(self.WIDTH):
            for y in range(self.HEIGHT):
                if (x, y) not in self.blocked_tiles:
                    self.all_nodes.append((x, y))

        # here, add start as source for net 0, add component sink as sink for net 0

        # then, add component source as source for net 1, add goal as sink for net 1

        # thus for add_flow_constraint, i need to pass in list of component source, a list of component sink, and try to connect them

        # component is used means the source and sink of it is also used.

        # pass in potential_component_whose_source_is_at_this_node
        # pass in potential_component_whose_sink_is_at_this_node

        # each potential sources have corresponding used flag
        
        self.jump_distances = jump_distances
        self.timelimit = timelimit
        self.symmetry = symmetry
        self.use_option = use_option

        # all possible location and orientation to place the components
        self.all_components: List[Component] = []
        self.node_related_components: Dict[Node, List[Component]] = defaultdict(list) # to record occupancy
        self.node_related_secondary_components: Dict[Node, List[Component]] = defaultdict(list) # to record occupancy of secondary component
        self.node_related_component_sources: Dict[Node, List[Component]] = defaultdict(list) # to record source
        self.node_related_component_secondary_sources: Dict[Node, List[Component]] = defaultdict(list) # to record secondary source
        self.node_related_component_sinks: Dict[Node, List[Component]] = defaultdict(list) # to record sink
        for node in self.all_nodes:
            x, y = node
            for dx, dy in DIRECTIONS:
                # compute secondary direction: direction but exclude the opposite direction and the same direction
                secondary_direction = DIRECTIONS.copy()
                secondary_direction.remove((dx, dy))
                secondary_direction.remove((-dx, -dy))
                
                for secondary_dx, secondary_dy in secondary_direction:
                    component = ((x, y), (dx, dy), (secondary_dx, secondary_dy))

                    # input location (sink)
                    ix = x - dx
                    iy = y - dy

                    # output location (source)
                    ox1 = x + dx
                    oy1 = y + dy
                    ox2 = x + secondary_dx + dx
                    oy2 = y + secondary_dy + dy

                    # skip if input location is invalid
                    if (ix, iy) not in self.all_nodes:
                        continue

                    # skip if output location is invalid
                    if (ox1, oy1) not in self.all_nodes or (ox2, oy2) not in self.all_nodes:
                        continue

                    self.all_components.append(component)
                    self.node_related_components[(x, y)].append(component)
                    self.node_related_secondary_components[(x + secondary_dx, y + secondary_dy)].append(component)
                    self.node_related_component_sources[(ox1, oy1)].append(component)
                    self.node_related_component_secondary_sources[(ox2, oy2)].append(component)
                    self.node_related_component_sinks[(x, y)].append(component)

        # is component used
        self.is_component_used: Dict[Component, Var] = {}
        self.is_component_used = {component: self.model.addVar(name=f"component_{component}", vtype=GRB.BINARY) for component in self.all_components}
        
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

        
        
        # Dynamic variables
        self.is_edge_used: Dict[int, Dict[Edge, Var]] = {}
        for i in range(self.num_nets):
            self.is_edge_used[i] = {edge: self.model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY) for edge in self.all_edges}
        
        self.is_node_used_by_net: Dict[int, Dict[Node, Var]] = self.dynamic_compute_is_node_used_by()

        # Objective function
        self.add_objective()

        # Constraints
        self.add_constraints()

        # Solve
        self.solve()

        # Plot
        self.plot()

    def dynamic_compute_is_node_used_by(self):
        is_node_used_by_net: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for node in self.all_nodes:

                step_edges_from_node = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
                jump_edges_related_to_node = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]
                
                is_node_used_by_net[i][node] = self.model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)
                # self.model.addConstr(len(step_edges_from_node) * is_node_used_by_net[i][node] >= quicksum(step_edges_from_node) + len(step_edges_from_node) * quicksum(jump_edges_related_to_node))
                self.model.addConstr(is_node_used_by_net[i][node] >= quicksum(step_edges_from_node) / len(step_edges_from_node) + quicksum(jump_edges_related_to_node))
        return is_node_used_by_net

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
            self.add_overlap_and_one_jump_constraints(i)
            self.add_directional_constraints_w_component(i)

        # self.add_goal_action_constraints()
        self.add_things_overlap_constraints()

        # self.add_component_count_constraint()

        if self.symmetry:
            self.add_symmetry_constraints()

    def add_flow_constraints(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype = GRB.INTEGER, lb=0, ub=4) for edge in self.all_edges}
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addConstr(self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * 4)
            self.model.addConstr(self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge])

        # Flow conservation constraints
        for node in self.all_nodes:
            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)

            if node in self.net_sources[i]:
                self.model.addConstr(in_flow == 0)
                self.model.addConstr(out_flow == 4)
            elif node in self.net_sinks[i]:
                self.model.addConstr(in_flow == 4)
                self.model.addConstr(out_flow == 0)
            else:
                self.model.addConstr(sum(in_flow) <= 4)
                self.model.addConstr((out_flow - in_flow == 0))
    
    def add_flow_constraints_source_to_components(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=4) for edge in self.all_edges}
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addConstr(self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * 4)
            self.model.addConstr(self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge])

        sources = self.net_sources[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_components[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_sink = self.model.addVar(name = f"node_is_component_sink_{i}_{node}", vtype=GRB.BINARY)
            if node_component_used_bool_list:
                # OR trick
                self.model.addConstr(node_is_component_sink >= quicksum(node_component_used_bool_list) / len(node_component_used_bool_list))
                self.model.addConstr(node_is_component_sink <= quicksum(node_component_used_bool_list))
            else:
                self.model.addConstr(node_is_component_sink == 0)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= 4)
            self.model.addConstr(out_flow <= 4)

            if node in sources:
                self.model.addConstr(in_flow == 0)
                self.model.addConstr(out_flow == 4)
            else:
                # Constraint 1: if z == 1 ⇒ in_flow == 4
                # IMPLICATION trick
                M = 4                    
                self.model.addConstr(in_flow >= self.component_sink_amount - M * (1 - node_is_component_sink))
                self.model.addConstr(in_flow <= self.component_sink_amount + M * (1 - node_is_component_sink))

                # Constraint 2: if z == 1 ⇒ out_flow == 0
                self.model.addConstr(out_flow <= 4 * (1 - node_is_component_sink))

                # Constraint 3: if z == 0 ⇒ in_flow == out_flow
                self.model.addConstr((out_flow - in_flow) <= node_is_component_sink * 4)
                self.model.addConstr((out_flow - in_flow) >= -node_is_component_sink * 4)
    
    def add_flow_constraints_component_to_goal(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=4) for edge in self.all_edges}
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addConstr(self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * 4)
            self.model.addConstr(self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge])

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_sources[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_source = self.model.addVar(name = f"node_is_component_source_{i}_{node}", vtype=GRB.BINARY)
            if node_component_used_bool_list:
                # OR trick
                self.model.addConstr(node_is_component_source >= quicksum(node_component_used_bool_list) / len(node_component_used_bool_list))
                self.model.addConstr(node_is_component_source <= quicksum(node_component_used_bool_list))
            else:
                self.model.addConstr(node_is_component_source == 0)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= 4)
            self.model.addConstr(out_flow <= 4)

            if node in sinks:
                self.model.addConstr(out_flow == 0)
                self.model.addConstr(in_flow == 4)
            else:
                # Constraint 1: if z == 1 ⇒ out_flow == component_source_amount * sum(node_component_used_bool_list)
                # IMPLICATION trick
                M = 4                    
                self.model.addConstr(out_flow >= self.component_source_amount * quicksum(node_component_used_bool_list) - M * (1 - node_is_component_source))
                self.model.addConstr(out_flow <= self.component_source_amount * quicksum(node_component_used_bool_list) + M * (1 - node_is_component_source))

                # Constraint 2: if z == 1 ⇒ in_flow == 0
                self.model.addConstr(in_flow <= 4 * (1 - node_is_component_source))

                # Constraint 3: if z == 0 ⇒ in_flow == out_flow
                self.model.addConstr((out_flow - in_flow) <= node_is_component_source * 4)
                self.model.addConstr((out_flow - in_flow) >= -node_is_component_source * 4)

    def add_flow_constraints_secondary_component_to_goal(self, i):
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = {}
        self.edge_flow_value[i] = {edge: self.model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=4) for edge in self.all_edges}
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            self.model.addConstr(self.edge_flow_value[i][edge] <= self.is_edge_used[i][edge] * 4)
            self.model.addConstr(self.edge_flow_value[i][edge] >= self.is_edge_used[i][edge])

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_secondary_sources[node]
            node_component_used_bool_list = [self.is_component_used[component] for component in node_components]
            node_is_component_source = self.model.addVar(name = f"node_is_component_source_{i}_{node}", vtype=GRB.BINARY)
            if node_component_used_bool_list:
                # OR trick
                self.model.addConstr(node_is_component_source >= quicksum(node_component_used_bool_list) / len(node_component_used_bool_list))
                self.model.addConstr(node_is_component_source <= quicksum(node_component_used_bool_list))
            else:
                self.model.addConstr(node_is_component_source == 0)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            self.model.addConstr(in_flow <= 4)
            self.model.addConstr(out_flow <= 4)

            if node in sinks:
                self.model.addConstr(out_flow == 0)
                self.model.addConstr(in_flow == 4)
            else:
                # Constraint 1: if z == 1 ⇒ out_flow == 4
                # IMPLICATION trick
                M = 4                    
                self.model.addConstr(out_flow >= self.component_source_amount * quicksum(node_component_used_bool_list) - M * (1 - node_is_component_source))
                self.model.addConstr(out_flow <= self.component_source_amount * quicksum(node_component_used_bool_list) + M * (1 - node_is_component_source))

                # Constraint 2: if z == 1 ⇒ in_flow == 0
                self.model.addConstr(in_flow <= 4 * (1 - node_is_component_source))

                # Constraint 3: if z == 0 ⇒ in_flow == out_flow
                self.model.addConstr((out_flow - in_flow) <= node_is_component_source * 4)
                self.model.addConstr((out_flow - in_flow) >= -node_is_component_source * 4)
                

    def add_flow_constraints_v2(self, i):
        for node in self.all_nodes:
            in_flow = [self.is_edge_used[i][edge] for edge in self.all_edges if edge[1] == node]
            out_flow = [self.is_edge_used[i][edge] for edge in self.all_edges if edge[0] == node]

            if node in self.net_sources[i]:
                self.model.addConstr(sum(in_flow) == 0)
                self.model.addConstr(sum(out_flow) == 1)
            elif node in self.net_sinks[i]:
                self.model.addConstr(sum(in_flow) == 1)
                self.model.addConstr(sum(out_flow) == 0)
            else:
                # if have in_flow, then must have out_flow
                self.model.addConstr(sum(in_flow) / len(in_flow) <= sum(out_flow))
                # self.model.addConstr(sum(in_flow) <= sum(out_flow) * len(in_flow))
                # if have out_flow, then must have in_flow
                self.model.addConstr(sum(out_flow) / len(out_flow) <= sum(in_flow))
                # self.model.addConstr(sum(out_flow) <= sum(in_flow) * len(out_flow))

        # the flow in and flow out must not be cyclic
        max_level = self.WIDTH + self.HEIGHT  # rough upper bound for longest path
        node_level = {}

        for node in self.all_nodes:
            node_level[node] = self.model.addVar(name=f"node_level_{node}", vtype=GRB.INTEGER)
                
        # Acyclic constraint using topological levels
        M = max_level + 1
        for edge in self.all_edges:
            u, v, _ = edge
            self.model.addConstr(node_level[v] >= node_level[u] + 1 - M * (1 - self.is_edge_used[i][edge]))

    def add_directional_constraints(self, i):
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            # A jump from u to v in direction `direction` is only allowed
            # if there is incoming flow to u from the same direction.

            # if u is at start, then only up jump is allowed
            if u in self.net_sources[i]:
                if direction == (0, 1):
                    continue
                else:
                    self.model.addConstr(self.is_edge_used[i][jump_edge] == 0)
                    continue

            # Collect all edges (step and jump) that go into `u` from direction `direction`
            incoming_edges_in_same_direction = []
            for edge in self.all_edges:
                _, target, dir_step = edge
                if target == u and dir_step == direction:
                    incoming_edges_in_same_direction.append(self.is_edge_used[i][edge])

            # create a variable that sums up the incoming edges in the same direction
            sum_of_incoming_edge_in_same_direction = quicksum(incoming_edges_in_same_direction)

            # Enforce that jump flow is only allowed if incoming flow matches direction
            self.model.addConstr(self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction)

    def add_directional_constraints_v2(self, i):
        # for jump edge at start, only up direction is allowed
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                if direction == (0, 1):
                    continue
                else:
                    self.model.addConstr(self.is_edge_used[i][jump_edge] == 0)
                    continue

        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for edge in self.all_edges:
            u, v, direction = edge

            # if the edge is used, then the end node must not have starting jump edge at different direction, and must not have any landing jump edge
            for jump_edge in self.node_related_jump_edges[v]:
                u2, v2, jump_direction = jump_edge
                if u2 == v: # starting jump edge
                    if direction == jump_direction:
                        continue
                    else:
                        self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1)
                else:
                    self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true
                
    def add_directional_constraints_w_component(self, i):
        # no jump edge at start
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                self.model.addConstr(self.is_edge_used[i][jump_edge] == 0)
        
        # component direction constraint
        for node in self.all_nodes:
            # if a node has active component source and secondary source, only jump in the component direction is allowed
            for component in self.node_related_component_sources[node] + self.node_related_component_secondary_sources[node]:
                _, component_direction, _ = component
                for jump_edge in self.node_related_jump_edges[node]:
                    u, v, direction = jump_edge
                    if u == node:
                        if direction == component_direction:
                            continue
                        else:
                            self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)
                    else:
                        self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)
            
            # if a node has active component sink, only jump in the component direction is allowed
            for component in self.node_related_component_sinks[node]:
                _, component_direction, _ = component
                related_jump_edge = [edge for edge in self.all_edges if edge[1] == node]
                for jump_edge in related_jump_edge:
                    u, v, direction = jump_edge
                    if v == node: # this is landing jump edge
                        if direction == component_direction:
                            continue
                        else:
                            self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)
                    else:
                        self.model.addConstr(self.is_edge_used[i][jump_edge] + self.is_component_used[component] <= 1)

        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for edge in self.all_edges:
            u, v, direction = edge

            # if the edge is used, then the end node must not have starting jump edge at different direction, and must not have any landing jump edge
            for jump_edge in self.node_related_jump_edges[v]:
                u2, v2, jump_direction = jump_edge
                if u2 == v: # starting jump edge
                    if direction == jump_direction:
                        continue
                    else:
                        self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1)
                else:
                    self.model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true
                

    def add_overlap_and_one_jump_constraints(self, i):
        for node in self.all_nodes:
            # list of all step edges from this node
            step_edges_from_node = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]

            # list of all jump edges of this node
            jump_edges_related_to_node = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]

            # the constraint
            self.model.addConstr(
                # quicksum(step_edges_from_node) + len(step_edges_from_node) * quicksum(jump_edges_related_to_node) <= len(step_edges_from_node)
                quicksum(step_edges_from_node) / len(step_edges_from_node) + quicksum(jump_edges_related_to_node) <= 1
            )

    # def add_overlap_constraints_v3(self):
    #     for node in self.all_nodes[0]:
    #         step_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)
    #         jump_edges_of_net: Dict[int, List[Edge]] = defaultdict(list)

    #         for i in range(self.num_nets):
    #             step_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
    #             jump_edges_of_net[i] = [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]

    #         # the constraint
    #         self.model.addConstr(()
    #             quicksum(quicksum(step_edges_of_net[i]) / len(step_edges_of_net[i]) for i in range(self.num_nets)) + 
    #             quicksum(quicksum(jump_edges_of_net[i]) for i in range(self.num_nets)) <= 1)

    # def add_overlap_constraints(self, i):
    #     for node in self.all_nodes:
    #         # Constraint: a node cannot be used by both step and jump
    #         self.model.addConstr(()
    #             self.is_node_used_by_step_edge[i][node] + self.is_node_used_by_jump_edge[i][node] <= 1
    #         )

    # def add_one_jump_constraints(self, i):
    #     # at most one jump edge allowed in each node
    #     for node in self.all_nodes:
    #         # create a variable that sums up the jump edges in the same node
    #         num_of_jump_edges_per_node = quicksum(self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node])

    #         # Enforce that at most one jump edge is allowed in each node
    #         self.model.addConstr(num_of_jump_edges_per_node <= 1)


    def add_goal_action_constraints(self):
        # no action is to be taken at the goal nodes for any net
        for i in range(self.num_nets):
            for j in range(self.num_nets):
                for goal in self.net_sinks[j]:
                    self.model.addConstr(self.is_node_used_by_net[i][goal] == 0)

    def add_net_overlap_constraints(self):
        # no overlap between nets
        for node in self.all_nodes:
            list_of_nets_using_node = []
            for i in range(self.num_nets):
                list_of_nets_using_node.append(self.is_node_used_by_net[i][node])
            
            # constraint: at most one net can use a node
            self.model.addConstr(quicksum(list_of_nets_using_node) <= 1)

    def add_component_overlap_constraints(self):
        # no overlap between components
        for node in self.all_nodes:
            list_of_components_bool_using_node = []
            for component in self.node_related_components[node]:
                list_of_components_bool_using_node.append(self.is_component_used[component])
            
            # constraint: at most one component can use a node
            self.model.addConstr(quicksum(list_of_components_bool_using_node) <= 1)

    def add_things_overlap_constraints(self):
        # between belts / pads / components
        for node in self.all_nodes:
            list_of_things_using_node = []
            for i in range(self.num_nets):
                list_of_things_using_node.append(self.is_node_used_by_net[i][node])
            for component in self.node_related_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            for component in self.node_related_secondary_components[node]:
                list_of_things_using_node.append(self.is_component_used[component])
            
            # constraint: at most one thing can use a node
            self.model.addConstr(quicksum(list_of_things_using_node) <= 1)

    def add_component_count_constraint(self):
        # add component count constraint
        component_used_bool_list = [self.is_component_used[component] for component in self.all_components]
        self.model.addConstr(quicksum(component_used_bool_list) == 2)

        # add component location constraint
        for component in self.all_components:
            # get the location of the component
            (x, y), (dx, dy), (dx2, dy2) = component

            if (x == 6 and y == 8 and dx == 0 and dy == 1 and dx2 == 1 and dy2 == 0):
                # add constraint
                self.model.addConstr(self.is_component_used[component] == 1            )

            if (x == 6 and y == 6 and dx == 0 and dy == -1 and dx2 == 1 and dy2 == 0):
                # add constraint
                self.model.addConstr(self.is_component_used[component] == 1            )

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
                sum_of_incoming_edge_in_same_direction = quicksum(incoming_edges_in_same_direction)

                # Enforce that jump pad flow is only allowed if incoming flow matches direction
                self.model.addConstr(self.is_edge_used[i][jump_edge] <= sum_of_incoming_edge_in_same_direction)

    def add_symmetry_constraints(self):
        # net i should reflex net K-i
        for i in range(int(self.num_nets / 2)):
            j = self.num_nets - i - 1
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
                    self.model.addConstr(self.is_edge_used[i][edge] == self.is_edge_used[j][(sym_u, sym_v, sym_d)])

    def solve(self):
        if self.timelimit != -1:
            self.model.setParam('TimeLimit', self.timelimit)
        if self.use_option:
            self.model.setParam('MIPFocus', 1)  # equivalent to use_option
        self.model.optimize()

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
        handle_component_square = Line2D([], [], marker='s', color='grey', markersize=14, markeredgecolor='black', linestyle='None')
        handle_component_circle = Line2D([], [], marker='o', color='grey', markersize=13, markeredgecolor='black', linestyle='None')
        handle_component = (handle_component_square, handle_component_circle)
        legend_handles = [handle_start, handle_jump_pad, handle_belt, handle_component]
        legend_labels  = ['Start/Goal', 'Jump Pad', 'Belt', 'Component']
        ax.legend(legend_handles, legend_labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc='upper right')

        # show
        plt.show()


# Example usage
if __name__ == "__main__":
    # nets = [
    #     # ([(5, 0)], 
    #     #  [(1, 6), (3, 6), (5, 6), (8, 6)]),
    #     # ([(28, 0)], 
    #     #  [(26, 6), (28, 6), (30, 6), (32, 6)]),

    #     # ([(5, 0), (6, 0), (7, 0), (8, 0)], 
    #     #  [(1, 6), (3, 6), (5, 6), (7, 6), (9, 6), (11, 6), (13, 6), (15, 6), (17, 6), (19, 6), (21, 6), (23, 6), (25, 6), (27, 6), (29, 6), (31, 6)]),
    #     # ([(25, 0), (26, 0), (27, 0), (28, 0)], 
    #     #  [(2, 6), (4, 6), (6, 6), (8, 6), (10, 6), (12, 6), (14, 6), (16, 6), (18, 6), (20, 6), (22, 6), (24, 6), (26, 6), (28, 6), (30, 6), (32, 6)]),

    #     ([(5, 0), (6, 0), (7, 0), (8, 0)], 
    #      [(1, 6), (4, 6), (5, 6), (8, 6), (9, 6), (12, 6), (13, 6), (16, 6), (17, 6), (20, 6), (21, 6), (24, 6), (25, 6), (28, 6), (29, 6), (32, 6)]),
    #     ([(25, 0), (26, 0), (27, 0), (28, 0)], 
    #      [(2, 6), (3, 6), (6, 6), (7, 6), (10, 6), (11, 6), (14, 6), (15, 6), (18, 6), (19, 6), (22, 6), (23, 6), (26, 6), (27, 6), (30, 6), (31, 6)]),
    #     ]
    # router = DirectionalJumpRouter(width=34, height=7, nets=nets, jump_distances= [4], timelimit = 300, symmetry = False)


    nets = [        
        # ([(6, 0)], 
        #  [(6, 15)]),

        ([(6, 0)], 
        [(6, 15)],
        [(0, 6)]),

        # ([(6, 0), (7, 0), (8, 0), (9, 0)], 
        #  [(6, 15), (7, 15), (8, 15), (9, 15)],
        #  [(0, 6), (0, 7), (0, 8), (0, 9)]),

        # ([(6, 0), (7, 0), (8, 0)], 
        #  [(6, 15), (7, 15), (8, 15)],
        #  [(0, 6), (0, 7), (0, 8)]),

        # ([(6, 0), (7, 0)], 
        #  [(6, 15), (7, 15)],
        #  [(0, 6), (0, 7)]),
        ]
    router = DirectionalJumpRouter(width=16, height=16, nets=nets, jump_distances= [1, 2, 3, 4], timelimit = 300, symmetry = False, use_option=True)


    # 169 cost without variable length launcher for 8 to 32 routing
    # # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/564a50812cc64b7f96e235e5bd1d7f8e-pulp.sol

    # 40 cost for cutter in 1x1
    # /var/folders/9f/zk98sm354s964kr7thw5t0980000gn/T/cab48f13fec049babcf0b53d7f8d3c99-pulp.sol