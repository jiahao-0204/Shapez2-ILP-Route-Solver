from Components.Component import Component
from constants import OFFSET, Node, Direction, Amount
import matplotlib.pyplot as plt
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class StartComponent(Component):
    def __init__(self, node: Node, direction: Direction, amount: Amount):
        super().__init__()

        # node and direction
        self.node = node
        self.direction = direction
        self.amount = amount
        self.source_node = (self.node[0] + self.direction[0], self.node[1] + self.direction[1])

        # x and y
        self.x, self.y = self.node
        self.nx, self.ny = self.source_node

    def draw(self, ax: plt.Axes):
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='s', s=120, edgecolors='black', zorder=0)
        ax.scatter(self.x + OFFSET, self.y + OFFSET, c=self.color, marker='o', s=50, edgecolors='black', zorder=2)
        ax.plot([self.x + OFFSET, self.nx + OFFSET], [self.y + OFFSET, self.ny + OFFSET], c='black', zorder=1)

    def add_constraints(self, router: "Router"):
        # null node
        null_node = self.node
        router.add_null_node_constraints(null_node)

        # source node
        source_node = (self.node[0] + self.direction[0], self.node[1] + self.direction[1])
        router.add_source_node_constraints(source_node, self.direction)

    def get_io_for_net(self):
        return (self, self.source_node, self.amount)