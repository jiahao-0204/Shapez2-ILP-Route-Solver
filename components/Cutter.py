from components.Component import Component
from constants import OFFSET, Node, Direction, Amount
import matplotlib.pyplot as plt
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class CutterComponent(Component):
    def __init__(self, node: Node, direction: Direction, secondary_direction: Direction, amount: Amount):
        super().__init__()
        
        # node and direction
        self.node = node
        self.direction = direction
        self.secondary_direction = secondary_direction
        self.amount = amount
        self.secondary_node = (self.node[0] + self.secondary_direction[0], self.node[1] + self.secondary_direction[1])
        self.primary_source = (self.node[0] + self.direction[0], self.node[1] + self.direction[1])
        self.secondary_source = (self.secondary_node[0] + self.direction[0], self.secondary_node[1] + self.direction[1])
        self.input_node = (self.node[0] - self.direction[0], self.node[1] - self.direction[1])

        # x and y
        self.x, self.y = self.node
        self.dx, self.dy = self.direction
        self.x2, self.y2 = self.secondary_node
        self.dx2, self.dy2 = self.secondary_direction
        self.nx, self.ny = self.primary_source
        self.nx2, self.ny2 = self.secondary_source

    def draw(self, ax: plt.Axes):
        # compute cutter dimensions
        margin = 0.2
        ll_x = min(self.x, self.x2) + margin
        ll_y = min(self.y, self.y2) + margin
        width  = abs(self.x2 - self.x) + 1 - 2 * margin       # +1 because each node is 1×1
        height = abs(self.y2 - self.y) + 1 - 2 * margin

        # draw rectangle
        rect = plt.Rectangle((ll_x, ll_y), width, height, facecolor='grey', edgecolor='black', linewidth=1.2, zorder=1)
        ax.add_patch(rect)

        # draw triangles
        if self.direction == (0, 1):
            marker = '^'
        elif self.direction == (0, -1):
            marker = 'v'
        elif self.direction == (1, 0):
            marker = '>'
        elif self.direction == (-1, 0):
            marker = '<'
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)
        ax.scatter(self.x2 + OFFSET, self.y2 + OFFSET, c='grey', marker=marker, s=80, edgecolors='black', zorder = 2)

        # draw lines
        ax.plot([self.x + OFFSET, self.nx + OFFSET], [self.y + OFFSET, self.ny + OFFSET], c='black', zorder=0)
        ax.plot([self.x2 + OFFSET, self.nx2 + OFFSET], [self.y2 + OFFSET, self.ny2 + OFFSET], c='black', zorder=0)

    def add_constraints(self, router: "Router"):
        router.add_sink_node_constraints(self.node, self.input_node, self.direction)
        router.add_null_node_constraints(self.secondary_node)
        router.add_source_node_constraints(self.primary_source, self.direction)
        router.add_source_node_constraints(self.secondary_source, self.direction)

    def get_io(self, ith):
        if ith == 0:
            return (self, self.node, self.amount)
        elif ith == 1:
            return (self, self.primary_source, self.amount)
        elif ith == 2:
            return (self, self.secondary_source, self.amount)
        else:
            raise ValueError("Invalid index for get_io")
    
    def get_nodes(self):
        return [self.node, self.secondary_node]