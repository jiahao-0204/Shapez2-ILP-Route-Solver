# Copyright (c) 2025 Jiahao
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from components.Component import Component
from constants import OFFSET, Node, Direction, Amount
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class SwapperComponent(Component):
    def __init__(self, node: Node, direction: Direction, secondary_direction: Direction, amount: Amount):
        super().__init__()
        
        # node and direction
        self.node1 = node
        self.direction = direction
        self.secondary_direction = secondary_direction
        self.amount = amount
        self.node2 = (self.node1[0] + self.secondary_direction[0], self.node1[1] + self.secondary_direction[1])
        self.source1 = (self.node1[0] + self.direction[0], self.node1[1] + self.direction[1])
        self.source2 = (self.node2[0] + self.direction[0], self.node2[1] + self.direction[1])
        self.input1 = (self.node1[0] - self.direction[0], self.node1[1] - self.direction[1])
        self.input2 = (self.node2[0] - self.direction[0], self.node2[1] - self.direction[1])

        # x and y
        self.x, self.y = self.node1
        self.dx, self.dy = self.direction
        self.x2, self.y2 = self.node2
        self.dx2, self.dy2 = self.secondary_direction
        self.nx, self.ny = self.source1
        self.nx2, self.ny2 = self.source2

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
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.io_color[self.node1], marker=marker, s=80, edgecolors='black', zorder = 2)
        ax.scatter(self.x2 + OFFSET, self.y2 + OFFSET, c=self.io_color[self.node2], marker=marker, s=80, edgecolors='black', zorder = 2)

        # draw lines
        ax.plot([self.x + OFFSET, self.nx + OFFSET], [self.y + OFFSET, self.ny + OFFSET], c='black', zorder=0)
        ax.plot([self.x2 + OFFSET, self.nx2 + OFFSET], [self.y2 + OFFSET, self.ny2 + OFFSET], c='black', zorder=0)

    def add_constraints(self, router: "Router"):
        router.add_sink_node_constraints(self.node1, self.input1, self.direction)
        router.add_sink_node_constraints(self.node2, self.input2, self.direction)
        router.add_source_node_constraints(self.source1, self.direction)
        router.add_source_node_constraints(self.source2, self.direction)

    def get_io(self, ith):
        if ith == 0:
            return (self, self.node1, self.amount)
        elif ith == 1:
            return (self, self.node2, self.amount)
        elif ith == 2:
            return (self, self.source1, self.amount)
        elif ith == 3:
            return (self, self.source2, self.amount)
        else:
            raise ValueError("Invalid index for get_io")
    
    def get_nodes(self):
        return [self.node1, self.node2]