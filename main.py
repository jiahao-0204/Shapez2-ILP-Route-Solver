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

    starts: List[Tuple[Node, Direction]] = []
    starts.append(((7, 0), (0, 1)))
    starts.append(((8, 0), (0, 1)))
    starts.append(((9, 0), (0, 1)))

    goals1: List[Tuple[Node, Direction]] = []
    goals1.append(((0, 7), (-1, 0)))
    goals1.append(((0, 8), (-1, 0)))
    goals1.append(((0, 9), (-1, 0)))

    goals2: List[Tuple[Node, Direction]] = []
    goals2.append(((7, 15), (0, 1)))
    goals2.append(((8, 15), (0, 1)))
    goals2.append(((9, 15), (0, 1)))

    cutter_list: List[Tuple[Node, Direction, Direction]] = []
    cutter_list.append(((4, 3), (1, 0), (0, 1)))
    cutter_list.append(((4, 7), (1, 0), (0, -1)))
    cutter_list.append(((4, 9), (1, 0), (0, 1)))
    # cutter_used.append(((4, 13), (1, 0), (0, -1)))

    cutter_list.append(((6, 3), (-1, 0), (0, 1)))
    cutter_list.append(((6, 7), (-1, 0), (0, -1)))
    cutter_list.append(((6, 9), (-1, 0), (0, 1)))
    # cutter_used.append(((6, 13), (-1, 0), (0, -1)))

    cutter_list.append(((9, 3), (1, 0), (0, 1)))
    cutter_list.append(((9, 7), (1, 0), (0, -1)))
    cutter_list.append(((9, 9), (1, 0), (0, 1)))
    # cutter_used.append(((9, 13), (1, 0), (0, -1)))

    cutter_list.append(((11, 3), (-1, 0), (0, 1)))
    cutter_list.append(((11, 7), (-1, 0), (0, -1)))
    cutter_list.append(((11, 9), (-1, 0), (0, 1)))
    # cutter_used.append(((11, 13), (-1, 0), (0, -1)))
    
    router = Router()
    router.route_cutters(width, height, cutter_list, starts, goals1, goals2, jump_distances, time_limit, option)