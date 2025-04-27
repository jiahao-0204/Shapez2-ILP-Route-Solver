import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.Start import StartComponent
from components.Goal import GoalComponent
from components.Swapper import SwapperComponent
from Router import Router
from constants import *
from typing import List

if __name__ == "__main__":
    starts1: List[StartComponent] = [
        StartComponent((6, 0), (0, 1), 4),
        StartComponent((7, 0), (0, 1), 4),
        StartComponent((8, 0), (0, 1), 4),
        StartComponent((9, 0), (0, 1), 4),
    ]

    starts2: List[StartComponent] = [
        StartComponent((26, 0), (0, 1), 4),
        StartComponent((27, 0), (0, 1), 4),
        StartComponent((28, 0), (0, 1), 4),
        StartComponent((29, 0), (0, 1), 4),
    ]
    
    goals1: List[GoalComponent] = [
        GoalComponent((6, 15), (0, 1), 4),
        GoalComponent((7, 15), (0, 1), 4),
        GoalComponent((8, 15), (0, 1), 4),
        GoalComponent((9, 15), (0, 1), 4),
    ]

    goals2: List[GoalComponent] = [
        GoalComponent((26, 15), (0, 1), 4),
        GoalComponent((27, 15), (0, 1), 4),
        GoalComponent((28, 15), (0, 1), 4),
        GoalComponent((29, 15), (0, 1), 4),
    ]

    swappers: List[SwapperComponent] = [
        SwapperComponent((2, 7), (0, 1), (1, 0), 1),
        SwapperComponent((5, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((6, 7), (0, 1), (1, 0), 1),
        SwapperComponent((9, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((10, 7), (0, 1), (1, 0), 1),
        SwapperComponent((13, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((14, 7), (0, 1), (1, 0), 1),
        SwapperComponent((17, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((18, 7), (0, 1), (1, 0), 1),
        SwapperComponent((21, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((22, 7), (0, 1), (1, 0), 1),
        SwapperComponent((25, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((26, 7), (0, 1), (1, 0), 1),
        SwapperComponent((29, 7), (0, 1), (-1, 0), 1),
        SwapperComponent((30, 7), (0, 1), (1, 0), 1),
        SwapperComponent((33, 7), (0, 1), (-1, 0), 1),
    ]
    
    # create router
    router = Router()
    router.initialize_board(width = 36, height = 16, jump_distances = [1, 2, 3, 4], num_nets = 4)
    router.add_components(starts1)
    router.add_components(starts2)
    router.add_components(goals1)
    router.add_components(goals2)
    router.add_components(swappers)
    router.generate_and_add_borders()
    router.add_net([c.get_io() for c in starts1], [c.get_io(0) for c in swappers])
    router.add_net([c.get_io() for c in starts2], [c.get_io(1) for c in swappers])
    router.add_net([c.get_io(2) for c in swappers], [c.get_io() for c in goals1])
    router.add_net([c.get_io(3) for c in swappers], [c.get_io() for c in goals2])
    router.draw()
    router.solve(timelimit = NO_TIME_LIMIT, option = MIPFOCUS_FEASIBILITY)