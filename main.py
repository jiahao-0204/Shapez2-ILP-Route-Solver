from Components.Start import StartComponent
from Components.Goal import GoalComponent
from Components.Cutter import CutterComponent
from Router import Router
from constants import *
from typing import List, Tuple

if __name__ == "__main__":
    # problem definition
    width = 16
    height = 16
    jump_distances = [1, 2, 3, 4]
    time_limit = NO_TIME_LIMIT
    option = MIPFOCUS_BALANCED
    # option = MIPFOCUS_BOUND

    starts: List[StartComponent] = []
    starts.append(StartComponent((7, 0), (0, 1), 4))
    starts.append(StartComponent((8, 0), (0, 1), 4))
    starts.append(StartComponent((9, 0), (0, 1), 4))

    goals1: List[GoalComponent] = []
    goals1.append(GoalComponent((0, 7), (-1, 0), 4))
    goals1.append(GoalComponent((0, 8), (-1, 0), 4))
    goals1.append(GoalComponent((0, 9), (-1, 0), 4))

    goals2: List[Tuple[Node, Direction]] = []
    goals2.append(GoalComponent((7, 15), (0, 1), 4))
    goals2.append(GoalComponent((8, 15), (0, 1), 4))
    goals2.append(GoalComponent((9, 15), (0, 1), 4))

    cutter_list: List[Tuple[Node, Direction, Direction]] = []
    cutter_list.append(CutterComponent((4, 3), (1, 0), (0, 1), 1))
    cutter_list.append(CutterComponent((4, 7), (1, 0), (0, -1), 1))
    cutter_list.append(CutterComponent((4, 9), (1, 0), (0, 1), 1))
    # cutter_used.append(CutterComponent((4, 13), (1, 0), (0, -1), 1))

    cutter_list.append(CutterComponent((6, 3), (-1, 0), (0, 1), 1))
    cutter_list.append(CutterComponent((6, 7), (-1, 0), (0, -1), 1))
    cutter_list.append(CutterComponent((6, 9), (-1, 0), (0, 1), 1))
    # cutter_used.append(CutterComponent((6, 13), (-1, 0), (0, -1), 1))

    cutter_list.append(CutterComponent((9, 3), (1, 0), (0, 1), 1))
    cutter_list.append(CutterComponent((9, 7), (1, 0), (0, -1), 1))
    cutter_list.append(CutterComponent((9, 9), (1, 0), (0, 1), 1))
    # cutter_used.append(CutterComponent((9, 13), (1, 0), (0, -1), 1))

    cutter_list.append(CutterComponent((11, 3), (-1, 0), (0, 1), 1))
    cutter_list.append(CutterComponent((11, 7), (-1, 0), (0, -1), 1))
    cutter_list.append(CutterComponent((11, 9), (-1, 0), (0, 1), 1))
    # cutter_used.append(CutterComponent((11, 13), (-1, 0), (0, -1), 1))
    
    router = Router()
    router.route_cutters(width, height, cutter_list, starts, goals1, goals2, jump_distances, time_limit, option)