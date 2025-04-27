from Components.Component import Component
from constants import OFFSET, Node, Direction, Amount
import matplotlib.pyplot as plt
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class GoalComponent(Component):
    def __init__(self, node: Node, direction: Direction, amount: Amount):
        super().__init__()

        # node and direction
        self.node = node
        self.direction = direction
        self.amount = amount
        self.input_node = (self.node[0] - self.direction[0], self.node[1] - self.direction[1])

        # x and y
        self.x, self.y = self.node
        self.ix, self.iy = self.input_node

    def draw(self, ax: plt.Axes):
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='s', s=120, edgecolors='black', zorder=0)
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='o', s=50, edgecolors='black', zorder=2)

    def add_constraints(self, router: "Router"):
        # as sink node
        router.add_sink_node_constraints(self.node, self.input_node, self.direction)

    def get_io(self, ith: int = 0):
        return (self, self.node, self.amount)