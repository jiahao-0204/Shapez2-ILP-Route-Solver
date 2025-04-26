from Components.Component import Component
from constants import OFFSET, Node, Direction
import matplotlib.pyplot as plt

class StartComponent(Component):
    def __init__(self, node: Node, direction: Direction, color: str):
        super().__init__()
        self.node = node
        self.direction = direction
        self.color = color

    def draw(self, ax: plt.Axes):
        x = self.node[0]
        y = self.node[1]
        nx = x + self.direction[0]
        ny = y + self.direction[1]

        ax.scatter(x + OFFSET, y + OFFSET, c=self.color, marker='s', s=120, edgecolors='black', zorder=0)
        ax.scatter(x + OFFSET, y + OFFSET, c=self.color, marker='o', s=50, edgecolors='black', zorder=2)
        ax.plot([x + OFFSET, nx + OFFSET], [y + OFFSET, ny + OFFSET], c='black', zorder=1)

    def add_constraints(self, solver):
        # null node
        null_node = self.node
        solver.add_null_node_constraints(null_node)

        # source node
        source_node = (self.node[0] + self.direction[0], self.node[1] + self.direction[1])
        solver.add_source_node_constraints(source_node, self.direction)
