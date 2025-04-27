# Copyright (c) 2025 Jiahao
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.Start import StartComponent
from components.Goal import GoalComponent
from components.Rotator import RotatorComponent
from Router import Router
from constants import *
from typing import List

if __name__ == "__main__":
    starts: List[StartComponent] = [
        StartComponent((6, 0), (0, 1), 4),
        StartComponent((7, 0), (0, 1), 4),
        StartComponent((8, 0), (0, 1), 4),
        StartComponent((9, 0), (0, 1), 4),
    ]
    
    goals: List[GoalComponent] = [
        GoalComponent((6, 15), (0, 1), 4),
        GoalComponent((7, 15), (0, 1), 4),
        GoalComponent((8, 15), (0, 1), 4),
        GoalComponent((9, 15), (0, 1), 4),
    ]
    
    rotators: List[RotatorComponent] = [
        RotatorComponent((4, 7), (0, 1), 2),
        RotatorComponent((5, 7), (0, 1), 2),
        RotatorComponent((6, 7), (0, 1), 2),
        RotatorComponent((7, 7), (0, 1), 2),
        RotatorComponent((8, 7), (0, 1), 2),
        RotatorComponent((9, 7), (0, 1), 2),
        RotatorComponent((10, 7), (0, 1), 2),
        RotatorComponent((11, 7), (0, 1), 2),
    ]
    
    # create router
    router = Router()
    router.initialize_board(width = 16, height = 16, jump_distances = [1, 2, 3, 4], num_nets = 3)
    router.add_components(starts)
    router.add_components(goals)
    router.add_components(rotators)
    router.generate_and_add_borders()
    router.add_net([c.get_io() for c in starts], [c.get_io(0) for c in rotators])
    router.add_net([c.get_io(1) for c in rotators], [c.get_io() for c in goals])
    router.solve(timelimit = NO_TIME_LIMIT, option = MIPFOCUS_FEASIBILITY, live_draw=True)