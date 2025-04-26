import gurobipy as gp
from gurobipy import GRB, quicksum

# Sample data
facilities = range(3)
customers  = range(5)
open_cost  = [5, 6, 4]        # cost to open each facility
assign_cost = [                # cost[j][i]: customer j served by facility i
    [2, 4, 5],
    [3, 1, 3],
    [4, 3, 2],
    [6, 2, 4],
    [3, 5, 3],
]

class BendersFacility:
    def __init__(self):
        self.build_master()
        self.master.Params.LazyConstraints = 1
        self.master.optimize(self.benders_callback)
        self.report()

    def build_master(self):
        self.master = gp.Model("master")

        # 1) Binary open[i]
        self.open = self.master.addVars(facilities, vtype=GRB.BINARY, name="open")

        # 2) Must open at least one facility so subproblem can be feasible
        self.master.addConstr(quicksum(self.open[i] for i in facilities) >= 1, name="AtLeastOneOpen")

        # 3) Auxiliary θ for assignment cost
        self.theta = self.master.addVar(lb=0.0, name="θ")

        # 4) Master objective: open cost + θ
        self.master.setObjective(quicksum(open_cost[i] * self.open[i] for i in facilities) + self.theta, GRB.MINIMIZE )

    def benders_callback(self, model, where):
        if where == GRB.Callback.MIPSOL:
            # 1) Get current open[] and θ
            open_vals = {i: model.cbGetSolution(self.open[i]) for i in facilities}
            theta_val = model.cbGetSolution(self.theta)
            # 2) Solve subproblem
            feas, cost = self.solve_subproblem(open_vals)
            if not feas:
                # Feasibility: ban this exact open-pattern
                used = [i for i,v in open_vals.items() if v > 0.5]
                expr = quicksum(self.open[i] for i in used)
                model.cbLazy(expr <= len(used) - 1)
            elif cost > theta_val + 1e-6:
                # Optimality: θ ≥ true assignment cost
                model.cbLazy(self.theta >= cost)

    def solve_subproblem(self, open_vals):
        """Given open[i]=0/1, solve the assignment subproblem."""
        sub = gp.Model("sub")
        sub.Params.OutputFlag = 0  # silent

        # 1) Fix open in this subproblem
        # 2) Assignment variables y[j,i] ∈ [0,1]
        y = sub.addVars(customers, facilities, lb=0.0, ub=1.0, name="y")
        # 3) Each customer fully assigned
        sub.addConstrs((quicksum(y[j,i] for i in facilities) == 1
                        for j in customers), name="assign")
        # 4) Cannot assign to unopened facilities
        sub.addConstrs((y[j,i] <= open_vals[i]
                        for j in customers for i in facilities),
                       name="open_link")
        # 5) Objective: minimize assignment cost
        sub.setObjective(
            quicksum(assign_cost[j][i] * y[j,i]
                     for j in customers for i in facilities),
            GRB.MINIMIZE
        )

        sub.optimize()
        if sub.Status == GRB.INFEASIBLE:
            return False, None
        else:
            return True, sub.ObjVal

    
    def report(self):
        print("\n*** Solution ***")
        if self.master.Status == GRB.OPTIMAL:
            print(f" Total cost = {self.master.ObjVal:.2f}")
            print(" Open facilities:")
            for i in facilities:
                if self.open[i].X > 0.5:
                    print(f"   Facility {i} (open_cost={open_cost[i]})")
            print(f" Assignment cost θ = {self.theta.X:.2f}")
        else:
            print(" No optimal solution found.")

if __name__ == "__main__":
    BendersFacility()
