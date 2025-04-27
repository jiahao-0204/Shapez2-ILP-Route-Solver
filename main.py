# Copyright (c) 2025 Jiahao
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from components.Start import StartComponent
from components.Goal import GoalComponent
from components.Cutter import CutterComponent
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
    
    goals1: List[GoalComponent] = [
        GoalComponent((0, 6), (-1, 0), 4),
        GoalComponent((0, 7), (-1, 0), 4),
        GoalComponent((0, 8), (-1, 0), 4),
        GoalComponent((0, 9), (-1, 0), 4),
    ]

    goals2: List[GoalComponent] = [
        GoalComponent((15, 6), (1, 0), 4),
        GoalComponent((15, 7), (1, 0), 4),
        GoalComponent((15, 8), (1, 0), 4),
        GoalComponent((15, 9), (1, 0), 4),
    ]
    
    cutters: List[CutterComponent] = [
        CutterComponent((4, 3), (1, 0), (0, 1), 1),
        CutterComponent((4, 7), (1, 0), (0, -1), 1),
        CutterComponent((4, 9), (1, 0), (0, 1), 1),
        CutterComponent((4, 13), (1, 0), (0, -1), 1),
        CutterComponent((6, 3), (-1, 0), (0, 1), 1),
        CutterComponent((6, 7), (-1, 0), (0, -1), 1),
        CutterComponent((6, 9), (-1, 0), (0, 1), 1),
        CutterComponent((6, 13), (-1, 0), (0, -1), 1),
        CutterComponent((9, 3), (1, 0), (0, 1), 1),
        CutterComponent((9, 7), (1, 0), (0, -1), 1),
        CutterComponent((9, 9), (1, 0), (0, 1), 1),
        CutterComponent((9, 13), (1, 0), (0, -1), 1),
        CutterComponent((11, 3), (-1, 0), (0, 1), 1),
        CutterComponent((11, 7), (-1, 0), (0, -1), 1),
        CutterComponent((11, 9), (-1, 0), (0, 1), 1),
        CutterComponent((11, 13), (-1, 0), (0, -1), 1),
    ]
    
    # create router
    router = Router()
    router.initialize_board(width = 16, height = 16, jump_distances = [1, 2, 3, 4], num_nets = 3)
    router.add_components(starts)
    router.add_components(goals1)
    router.add_components(goals2)
    router.add_components(cutters)
    router.generate_and_add_borders()
    router.add_net([c.get_io() for c in starts], [c.get_io(0) for c in cutters])
    router.add_net([c.get_io(1) for c in cutters], [c.get_io() for c in goals1])
    router.add_net([c.get_io(2) for c in cutters], [c.get_io() for c in goals2])
    router.solve(timelimit = NO_TIME_LIMIT, option = MIPFOCUS_FEASIBILITY)