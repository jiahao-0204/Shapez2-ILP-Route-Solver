from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List, Set

STEP_COST = 1
JUMP_COST = 2
STARTING_PAD = 0
LANDING_PAD = 1

PAD_TYPE = int
Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Edge = Tuple[Node, Node, Direction] # start, end, direciton

class SubProblem:
    def __init__(self, net_sources, net_sinks, all_nodes, all_edges, step_edges, jump_edges, 
                 node_related_components,
                 node_related_secondary_components, node_related_component_sources,
                 node_related_component_secondary_sources, node_related_component_sinks, node_related_belt_edges,
                 node_related_starting_pad_edges, node_related_landing_pad_edges):
        
        # model parameters
        self.timelimit = -1
        self.flow_priority = 50
        self.edge_priority = 0
        self.option = 1

        # problem parameters
        self.num_nets = 3
        self.start_amount = 4
        self.goal_amount = 4
        self.component_count = 16
        self.total_start_amount = self.component_count * self.start_amount 
        self.total_goal_amount = self.component_count * self.goal_amount
        self.total_secondary_goal_amount = self.component_count * self.goal_amount
        self.component_source_amount = 1
        self.component_sink_amount = 1
        self.flow_cap = 4


        self.net_sources = net_sources
        self.net_sinks = net_sinks
        self.all_nodes = all_nodes
        self.all_edges = all_edges
        self.step_edges = step_edges
        self.jump_edges = jump_edges

        # edges that generate this belt / starting pad / landing pad
        self.node_related_belt_edges = node_related_belt_edges
        self.node_related_starting_pad_edges = node_related_starting_pad_edges
        self.node_related_landing_pad_edges = node_related_landing_pad_edges

        self.node_related_components = node_related_components
        self.node_related_secondary_components = node_related_secondary_components
        self.node_related_component_sources = node_related_component_sources
        self.node_related_component_secondary_sources = node_related_component_secondary_sources
        self.node_related_component_sinks = node_related_component_sinks
        

    def solve_subproblem(self, is_component_used):
        # set up model parameters
        sub_model = Model("subproblem")
        if self.timelimit != -1:
            sub_model.Params.TimeLimit = self.timelimit
        sub_model.Params.MIPFocus = 1
        sub_model.Params.Presolve = 2
        # sub_model.Params.OutputFlag = 0  # silent

        self.is_edge_used: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.edge_flow_value: Dict[int, Dict[Edge, Var]] = defaultdict(lambda: defaultdict(Var))
        self.node_in_flow_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.node_out_flow_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
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

                # in flow and out flow
                self.node_out_flow_expr[i][edge[0]].addTerms(1, self.edge_flow_value[i][edge])
                self.node_in_flow_expr[i][edge[1]].addTerms(1, self.edge_flow_value[i][edge])
        self.compute_is_edge_used(sub_model)
        self.compute_is_node_used_by_belt(sub_model)
        self.add_flow_max_value_constraints(sub_model)
        self.add_belt_pad_net_overlap_constraints(sub_model)
        self.add_pad_direction_constraints(sub_model)
        
        # start and goals
        starts: List[Tuple[Node, Direction]] = []
        goals1: List[Tuple[Node, Direction]] = []
        goals2: List[Tuple[Node, Direction]] = []
        for node in self.net_sources[0]:
            starts.append((node, (0, 1)))
        for node in self.net_sinks[1]:
            goals1.append((node, (-1, 0)))
        for node in self.net_sinks[2]:
            goals2.append((node, (0, 1)))

        # cutters
        cutters = [compoenent for compoenent, value in is_component_used.items() if value > 0.5]

        # general
        self.add_objective(sub_model)
        
        # add start and goal
        self.add_start_edge_constraints(sub_model, starts)
        self.add_goal_edge_constraints(sub_model, goals1)
        self.add_goal_edge_constraints(sub_model, goals2)

        # add cutter
        self.add_cutter_edge_constraints(sub_model, cutters)
        self.add_cutter_net(sub_model, cutters)

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
                sub_model.addConstr(self.node_in_flow_expr[i][node] <= self.flow_cap)
                sub_model.addConstr(self.node_out_flow_expr[i][node] <= self.flow_cap)

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
            # null_node = node
            # source_node = (node[0] + direction[0], node[1] + direction[1])
            # self.add_null_node_constraints(sub_model, null_node)
            # self.add_source_node_constraints(sub_model, source_node, direction)

            # pass
            for i in range(self.num_nets):
                # no in flow
                in_flow_edges = [edge for edge in self.all_edges if edge[1] == node]
                for edge in in_flow_edges:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

                # out flow only by step and only in given direction
                out_flow_edges = self.node_related_belt_edges[node]
                for edge in out_flow_edges:
                    if edge[2] != direction:
                        sub_model.addConstr(self.is_edge_used[i][edge] == 0) 

                # no starting pad and landing pad
                for edge in self.node_related_starting_pad_edges[node] + self.node_related_landing_pad_edges[node]:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)
    
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

    def add_cutter_net(self, sub_model, cutters):
        # net 0: start -> componenent sink
        # net 1: component source -> goal
        # net 2: component secondary source -> goal
        s0 = self.net_sources[0]
        k0 = []
        s1 = []
        k1 = self.net_sinks[1]
        s2 = []
        k2 = self.net_sinks[2]

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
            in_flow = self.node_in_flow_expr[i][node]
            out_flow = self.node_out_flow_expr[i][node]

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
            in_flow_edges = [edge for edge in self.all_edges if edge[1] == node]
            for edge in in_flow_edges:
                if edge[2] == (-direction[0], -direction[1]):
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # outflow: except in opposite direction
            out_flow_edges = [edge for edge in self.all_edges if edge[0] == node]
            for edge in out_flow_edges:
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
            in_flow_edges = [edge for edge in self.all_edges if edge[1] == node]
            for edge in in_flow_edges:
                if edge[2] != direction:
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            out_flow_edges = [edge for edge in self.all_edges if edge[0] == node]
            for edge in out_flow_edges:
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
            in_flow_edges = [edge for edge in self.all_edges if edge[1] == node]
            for edge in in_flow_edges:
                sub_model.addConstr(self.is_edge_used[i][edge] == 0)

            # out flow: no
            out_flow_edges = [edge for edge in self.all_edges if edge[0] == node]
            for edge in out_flow_edges:
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