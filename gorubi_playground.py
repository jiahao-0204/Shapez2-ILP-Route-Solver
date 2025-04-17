import gurobipy as gp
from gurobipy import GRB

# Create a new model
model = gp.Model("minimal_ilp_example")

# Create variables
x = model.addVar(vtype=GRB.INTEGER, name="x")
y = model.addVar(vtype=GRB.INTEGER, name="y")

# Set objective
model.setObjective(3 * x + 2 * y, GRB.MAXIMIZE)

# Add constraints
model.addConstr(x + 2 * y <= 4, name="c1")
model.addConstr(3 * x + y <= 5, name="c2")

# Optimize the model
model.optimize()

# Print the solution
if model.status == GRB.OPTIMAL:
    print(f"Optimal solution: x = {x.X}, y = {y.X}, objective = {model.ObjVal}")
else:
    print("No optimal solution found.")
