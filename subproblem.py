from gurobipy import Model, GRB, quicksum, Var, LinExpr
from collections import defaultdict
from typing import Dict, Tuple, List

STEP_COST = 1
JUMP_COST = 2

Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Edge = Tuple[Node, Node, Direction] # start, end, direciton

class SubProblem:
    def __init__(self, net_sources, net_sinks, all_nodes, all_edges, step_edges, jump_edges, 
                 node_related_step_edges, node_related_jump_edges, node_related_components,
                 node_related_secondary_components, node_related_component_sources,
                 node_related_component_secondary_sources, node_related_component_sinks):
        
        # model parameters
        self.timelimit = -1
        self.edge_priority = 50
        self.flow_priority = 25
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
        self.node_related_step_edges = node_related_step_edges
        self.node_related_jump_edges = node_related_jump_edges
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
        self.is_node_used_by_step_edge: Dict[int, Dict[Node, Var]] = defaultdict(lambda: defaultdict(Var))
        self.node_in_flow_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))
        self.node_out_flow_expr: Dict[int, Dict[Node, LinExpr]] = defaultdict(lambda: defaultdict(LinExpr))

        # is edge used & edge flow value
        for i in range(self.num_nets):
            for edge in self.all_edges:

                # is edge used
                self.is_edge_used[i][edge] = sub_model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY)
                self.is_edge_used[i][edge].setAttr("BranchPriority", self.edge_priority)

                # edge flow value
                self.edge_flow_value[i][edge] = sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap)
                self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

                # constraint
                sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
                sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        # component occupied nodes
        component_occupied_nodes = set()
        self.component_source_node_and_direction: List[Tuple[Node, Direction]] = []
        self.component_input_node_and_direction: List[Tuple[Node, Direction]] = []
        for component, value in is_component_used.items():
            if value > 0.5:
                component_node, direction, secondary_direction = component
                secondary_component_node = (component_node[0] + secondary_direction[0], component_node[1] + secondary_direction[1])
                component_primary_source_node = (component_node[0] + direction[0], component_node[1] + direction[1])
                component_secondary_source_node = (secondary_component_node[0] + direction[0], secondary_component_node[1] + direction[1])
                component_input_node = (component_node[0] - direction[0], component_node[1] - direction[1])
                component_occupied_nodes.add(component_node)
                component_occupied_nodes.add(secondary_component_node)
                self.component_source_node_and_direction.append((component_primary_source_node, direction))
                self.component_source_node_and_direction.append((component_secondary_source_node, direction))
                self.component_input_node_and_direction.append((component_input_node, direction))
        # belts and jump pads at component occupied nodes
        for node in component_occupied_nodes:
            for edge in self.node_related_step_edges[node] + self.node_related_jump_edges[node]:
                for i in range(self.num_nets):
                    sub_model.addConstr(self.is_edge_used[i][edge] == 0)

        # is node used by step edge
        for i in range(self.num_nets):
            for node in self.all_nodes:
                node_step_edges_bool_list = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
                self.is_node_used_by_step_edge[i][node] = sub_model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)

                sub_model.addGenConstrOr(self.is_node_used_by_step_edge[i][node], node_step_edges_bool_list)

        # node in flow and out flow value
        for i in range(self.num_nets):
            for node in self.all_nodes:
                self.node_in_flow_expr[i][node] = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
                self.node_out_flow_expr[i][node] = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)

                sub_model.addConstr(self.node_in_flow_expr[i][node] <= self.flow_cap)
                sub_model.addConstr(self.node_out_flow_expr[i][node] <= self.flow_cap)

        # Objective function
        self.add_objective(sub_model)

        # Constraints
        self.add_constraints(sub_model, is_component_used)

        # Solve
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

    def add_constraints(self, sub_model, is_component_used):
        self.add_net_from_cutter_components(sub_model, is_component_used)
        self.add_things_overlap_constraints(sub_model)

        for i in range(self.num_nets):
            self.add_regular_directional_constraints(i, sub_model)
            self.add_component_directional_constraints(i, sub_model)

    def add_net_from_cutter_components(self, sub_model, is_component_used):
        # net 0: start -> componenent sink
        # net 1: component source -> goal
        # net 2: component secondary source -> goal
        s0 = self.net_sources[0]
        k0 = []
        s1 = []
        k1 = self.net_sinks[1]
        s2 = []
        k2 = self.net_sinks[2]

        for component, value in is_component_used.items():
            if value > 0.5:
                sink, direction, secondary_direction = component
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

    def add_regular_directional_constraints(self, i, sub_model):
        # no jump edge at start
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                sub_model.addConstr(self.is_edge_used[i][jump_edge] == 0)
        
        # for each edge, if the edge is used, then the end node must not have jump edge at different direction
        for edge in self.all_edges:
            u, v, direction = edge

            # if the edge is used, then the end node must not have starting jump edge at different direction, and must not have any landing jump edge
            for jump_edge in self.node_related_jump_edges[v]:
                u2, v2, jump_direction = jump_edge
                if u2 == v and direction == jump_direction: # starting jump edge
                    continue
                else:
                    # sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.is_edge_used[i][jump_edge] == 0)
                    sub_model.addConstr(self.is_edge_used[i][edge] + self.is_edge_used[i][jump_edge] <= 1) # only one can be true

    def add_component_directional_constraints(self, i, sub_model):
        # source location: can only have starting jump pad in the same direction
        for node, direction in self.component_source_node_and_direction:
            for jump_edge in self.node_related_jump_edges[node]:
                u, v, jump_direction = jump_edge
                if u == node and direction == jump_direction:
                    # skip if jump pad is allowed
                    continue
                sub_model.addConstr(self.is_edge_used[i][jump_edge] == 0)

        # input location: can only have landing jump pad in the same direction
        for node, direction in self.component_input_node_and_direction:
            for jump_edge in self.node_related_jump_edges[node]:
                u, v, jump_direction = jump_edge
                if u != node and direction == jump_direction:
                    # skip if jump pad is allowed
                    continue
                sub_model.addConstr(self.is_edge_used[i][jump_edge] == 0)

    def add_things_overlap_constraints(self, sub_model):
        # between belts / pads in different nets
        for node in self.all_nodes:
            list_of_things_using_node = []
            for i in range(self.num_nets):
                list_of_things_using_node += [self.is_node_used_by_step_edge[i][node]]
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]
            
            # constraint: at most one thing can use a node
            sub_model.addConstr(quicksum(list_of_things_using_node) <= 1)
            
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