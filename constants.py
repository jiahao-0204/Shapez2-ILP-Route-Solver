# constants.py
from typing import Tuple

# constants
STEP_COST = 1
JUMP_COST = 2
EDGE_PRIORITY = 50
FLOW_PRIORITY = 25

# flow 
FLOW_CAP = 4
IO_AMOUNT = 4
CUTTER_AMOUNT = 1

# pad types
PAD_TYPE = int
STARTING_PAD = 0
LANDING_PAD = 1

# directions
DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)] # right, up, left, down

# options
MIPFOCOUS_TYPE = int
MIPFOCUS_BALANCED = 0
MIPFOCUS_FEASIBILITY = 1
MIPFOCUS_OPTIMALITY = 2
MIPFOCUS_BOUND = 3
NO_TIME_LIMIT = -1

# solver settings
PRESOLVE = 2
HEURISTICS = 0.5

# types
Node = Tuple[int, int] # (x, y)
Direction = Tuple[int, int] # (dx, dy)
Edge = Tuple[Node, Node, Direction] # start, end, direciton
Amount = int
OFFSET = 0.5