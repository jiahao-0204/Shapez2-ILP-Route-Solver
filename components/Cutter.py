from Components.Component import Component
from constants import OFFSET, Node, Direction
import matplotlib.pyplot as plt
from typing import Tuple

class CutterComponent(Component):
    def __init__(self, cutter: Tuple[Node, Direction, Direction]):
        super().__init__()
        self.cutter = cutter

    def draw(self, ax: plt.Axes):
        # extract cutter information
        (x, y), (dx, dy), (dx2, dy2) = self.cutter
        x2, y2 = x + dx2, y + dy2
        nx, ny = x + dx, y + dy
        nx2, ny2 = x2 + dx, y2 + dy

        # compute cutter dimensions
        margin = 0.2
        ll_x = min(x, x2) + margin
        ll_y = min(y, y2) + margin
        width  = abs(x2 - x) + 1 - 2 * margin       # +1 because each node is 1×1
        height = abs(y2 - y) + 1 - 2 * margin

        # draw rectangle
        rect = plt.Rectangle((ll_x, ll_y), width, height, facecolor='grey', edgecolor='black', linewidth=1.2, zorder=1)
        ax.add_patch(rect)

        # draw triangles
        d = (dx, dy)
        if d == (0, 1):
            marker = '^'
        elif d == (0, -1):
            marker = 'v'
        elif d == (1, 0):
            marker = '>'
        elif d == (-1, 0):
            marker = '<'
        ax.scatter(x + OFFSET, y + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
        ax.scatter(x2 + OFFSET, y2 + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)

        # draw lines
        ax.plot([x + OFFSET, nx + OFFSET], [y + OFFSET, ny + OFFSET], c='black', zorder=0)
        ax.plot([x2 + OFFSET, nx2 + OFFSET], [y2 + OFFSET, ny2 + OFFSET], c='black', zorder=0)

    def add_constraints(self, solver):
        primary_component, direction, secondary_direction = self.cutter
        solver.add_sink_node_constraints(primary_component, direction)
        
        secondary_component = (primary_component[0] + secondary_direction[0], primary_component[1] + secondary_direction[1])
        solver.add_null_node_constraints(secondary_component)
        
        primary_source = (primary_component[0] + direction[0], primary_component[1] + direction[1])
        solver.add_source_node_constraints(primary_source, direction)
        
        secondary_source = (secondary_component[0] + direction[0], secondary_component[1] + direction[1])
        solver.add_source_node_constraints(secondary_source, direction)    
