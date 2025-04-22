from gurobipy import Model, GRB, quicksum, Var
from collections import defaultdict

STEP_COST = 1
JUMP_COST = 2

class SubProblem:
    def __init__(self, net_sources, net_sinks, all_nodes, all_edges, step_edges, jump_edges, 
                 node_related_step_edges, node_related_jump_edges, node_related_components,
                 node_related_secondary_components, node_related_component_sources,
                 node_related_component_secondary_sources, node_related_component_sinks):
        self.timelimit = -1
        self.num_nets = 3
        self.edge_priority = 50
        self.flow_priority = 25
        self.start_amount = 4
        self.goal_amount = 4
        self.component_count = 16
        self.total_start_amount = self.component_count * self.start_amount 
        self.total_goal_amount = self.component_count * self.goal_amount
        self.total_secondary_goal_amount = self.component_count * self.goal_amount
        self.component_source_amount = 1
        self.component_sink_amount = 1
        self.flow_cap = 4
        self.option = 1


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

        self.is_edge_used = {}
        self.is_node_used_by_step_edge = {}
        self.edge_flow_value = {}
        
        # is edge used
        for i in range(self.num_nets):
            self.is_edge_used[i] = {edge: sub_model.addVar(name=f"edge_{i}_{edge}", vtype=GRB.BINARY) for edge in self.all_edges}
            # set priority
            for edge in self.all_edges:
                self.is_edge_used[i][edge].setAttr("BranchPriority", self.edge_priority)
        
        self.add_variable_is_node_used_by_step_edges(sub_model)

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

    
    def add_variable_is_node_used_by_step_edges(self, sub_model):
        self.is_node_used_by_step_edge = defaultdict(lambda: defaultdict(Var))
        for i in range(self.num_nets):
            for node in self.all_nodes:
                node_step_edges_bool_list = [self.is_edge_used[i][edge] for edge in self.node_related_step_edges[node]]
                self.is_node_used_by_step_edge[i][node] = sub_model.addVar(name=f"node_{i}_{node}", vtype=GRB.BINARY)
                sub_model.addGenConstrOr(self.is_node_used_by_step_edge[i][node], node_step_edges_bool_list)

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
        self.add_flow_constraints_source_to_components(0, sub_model, is_component_used)
        self.add_flow_constraints_component_to_goal(1, sub_model, is_component_used)
        self.add_flow_constraints_secondary_component_to_goal(2, sub_model, is_component_used)

        for i in range(self.num_nets):
            # self.add_no_step_jump_overlap_constraints(i)
            self.add_directional_constraints_w_component(i, sub_model, is_component_used)

        self.add_things_overlap_constraints(sub_model, is_component_used)

    def add_flow_constraints_source_to_components(self, i, sub_model, is_component_used):
        self.edge_flow_value = {}
        self.edge_flow_value[i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)
        
        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sources = self.net_sources[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_components[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_sink = any(node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sources:
                sub_model.addConstr(in_flow == 0)
                start_amount = min(self.start_amount, self.total_start_amount)
                self.total_start_amount -= start_amount
                sub_model.addConstr(out_flow == start_amount)
            else:
                if node_is_component_sink:
                    sub_model.addConstr(in_flow - out_flow == self.component_sink_amount)
                else:
                    sub_model.addConstr(in_flow - out_flow == 0)
    
    def add_flow_constraints_component_to_goal(self, i, sub_model, is_component_used):
        self.edge_flow_value = {}
        self.edge_flow_value[i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_sources[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_source = any(node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                sub_model.addConstr(out_flow == 0)
                goal_amount = min(self.goal_amount, self.total_goal_amount)
                self.total_goal_amount -= goal_amount
                sub_model.addConstr(in_flow == goal_amount)
            else:
                if node_is_component_source:
                    sub_model.addConstr(out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))
                else:
                    sub_model.addConstr(out_flow - in_flow == 0)

    def add_flow_constraints_secondary_component_to_goal(self, i, sub_model, is_component_used):
        self.edge_flow_value = {}
        self.edge_flow_value[i] = {edge: sub_model.addVar(name = f"edge_flow_value_{i}_{edge}", vtype=GRB.INTEGER, lb=0, ub=self.flow_cap) for edge in self.all_edges}
        # set priority
        for edge in self.all_edges:
            self.edge_flow_value[i][edge].setAttr("BranchPriority", self.flow_priority)

        # Flow is avaiable if the edge is selected
        for edge in self.all_edges:
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], True, self.edge_flow_value[i][edge] >= 1)
            sub_model.addGenConstrIndicator(self.is_edge_used[i][edge], False, self.edge_flow_value[i][edge] == 0)

        sinks = self.net_sinks[i]

        # Flow conservation constraints for each net
        for node in self.all_nodes:
            node_components = self.node_related_component_secondary_sources[node]
            node_component_used_bool_list = [is_component_used[component] > 0.5 for component in node_components]
            node_is_component_source = any(node_component_used_bool_list)

            in_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[1] == node)
            out_flow = quicksum(self.edge_flow_value[i][edge] for edge in self.all_edges if edge[0] == node)
            
            # no matter what
            sub_model.addConstr(in_flow <= self.flow_cap)
            sub_model.addConstr(out_flow <= self.flow_cap)

            if node in sinks:
                sub_model.addConstr(out_flow == 0)
                goal_amount = min(self.goal_amount, self.total_secondary_goal_amount)
                self.total_secondary_goal_amount -= goal_amount
                sub_model.addConstr(in_flow == goal_amount)
            else:
                if node_is_component_source:
                    sub_model.addConstr(out_flow - in_flow == self.component_source_amount * quicksum(node_component_used_bool_list))
                else:
                    sub_model.addConstr(out_flow - in_flow == 0)
                  
    def add_directional_constraints_w_component(self, i, sub_model, is_component_used):
        # no jump edge at start
        for jump_edge in self.jump_edges:
            u, v, direction = jump_edge
            if u in self.net_sources[i]:
                sub_model.addConstr(self.is_edge_used[i][jump_edge] == 0)
        
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
                        # sub_model.addGenConstrIndicator(is_component_used[component], True, self.is_edge_used[i][jump_edge] == 0)
                        if is_component_used[component] > 0.5:
                            sub_model.addConstr(self.is_edge_used[i][jump_edge] == 0)
            
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
                        # sub_model.addGenConstrIndicator(is_component_used[component], True, self.is_edge_used[i][jump_edge] == 0)
                        if is_component_used[component] > 0.5:
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

    def add_things_overlap_constraints(self, sub_model, is_component_used):
        # between belts / pads / components
        for node in self.all_nodes:
            list_of_things_using_node = []
            occupied = False
            # for i in range(self.num_nets):
            #     list_of_things_using_node.append(self.is_node_used_by_net[i][node])
            for i in range(self.num_nets):
                list_of_things_using_node.append(self.is_node_used_by_step_edge[i][node])
                list_of_things_using_node += [self.is_edge_used[i][edge] for edge in self.node_related_jump_edges[node]]
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