from Components.Component import Component
from constants import OFFSET, Node, Direction
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class GoalComponent(Component):
    def __init__(self, node: Node, direction: Direction, color: str):
        super().__init__()
        self.node = node
        self.direction = direction
        self.color = color

        self.x = node[0]
        self.y = node[1]
        self.ix = self.x - direction[0]
        self.iy = self.y - direction[1]

        self.input_node = (self.ix, self.iy)

    def draw(self, ax: plt.Axes):
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='s', s=120, edgecolors='black', zorder=0)
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='o', s=50, edgecolors='black', zorder=2)

    def add_constraints(self, router: "Router"):
        # as sink node
        router.add_sink_node_constraints(self.node, self.input_node, self.direction)
