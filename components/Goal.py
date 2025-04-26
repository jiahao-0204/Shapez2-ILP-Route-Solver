from Components.Component import Component
from constants import OFFSET, Node, Direction
import matplotlib.pyplot as plt

class GoalComponent(Component):
    def __init__(self, node: Node, direction: Direction, color: str):
        super().__init__()
        self.node = node
        self.direction = direction
        self.color = color

    def draw(self, ax: plt.Axes):
        x = self.node[0]
        y = self.node[1]
        ix = x - self.direction[0]
        iy = y - self.direction[1]

        ax.scatter(x + OFFSET, y + OFFSET, c=self.color, marker='s', s=120, edgecolors='black', zorder=0)
        ax.scatter(x + OFFSET, y + OFFSET, c=self.color, marker='o', s=50, edgecolors='black', zorder=2)

    def add_constraints(self, solver):
        # as sink node
        solver.add_sink_node_constraints(self.node, self.direction)
