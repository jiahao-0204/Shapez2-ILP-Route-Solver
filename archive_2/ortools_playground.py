from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Create variables
x = model.NewIntVar(0, 10, "x")
y = model.NewIntVar(0, 10, "y")

# Add a constraint: x + y = 10
model.Add(x + y == 10)

# Objective: Maximize x
model.Maximize(x)

# Create solver and solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Output result
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Solution found: x = {solver.Value(x)}, y = {solver.Value(y)}")
else:
    print("No solution found.")
