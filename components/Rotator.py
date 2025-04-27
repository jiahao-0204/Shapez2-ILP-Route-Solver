from components.Component import Component
from constants import OFFSET, Node, Direction, Amount
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class RotatorComponent(Component):
    def __init__(self, node: Node, direction: Direction, amount: Amount):
        super().__init__()
        
        # node and direction
        self.node = node
        self.direction = direction
        self.amount = amount
        self.source_node = (self.node[0] + self.direction[0], self.node[1] + self.direction[1])
        self.input_node = (self.node[0] - self.direction[0], self.node[1] - self.direction[1])

        # x and y
        self.x, self.y = self.node
        self.dx, self.dy = self.direction
        self.nx, self.ny = self.source_node

    def draw(self, ax: plt.Axes):
        # compute cutter dimensions
        margin = 0.2
        ll_x = self.x + margin
        ll_y = self.y + margin
        width  = 1 - 2 * margin       # +1 because each node is 1×1
        height = 1 - 2 * margin

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

        # draw lines
        ax.plot([self.x + OFFSET, self.nx + OFFSET], [self.y + OFFSET, self.ny + OFFSET], c='black', zorder=0)

    def add_constraints(self, router: "Router"):
        router.add_sink_node_constraints(self.node, self.input_node, self.direction)
        router.add_source_node_constraints(self.source_node, self.direction)

    def get_io(self, ith):
        if ith == 0:
            return (self, self.node, self.amount)
        elif ith == 1:
            return (self, self.source_node, self.amount)
        else:
            raise ValueError("Invalid index for get_io")
    
    def get_nodes(self):
        return [self.node]