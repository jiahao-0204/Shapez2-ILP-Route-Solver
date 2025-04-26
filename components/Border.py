from Components.Component import Component
from constants import Node
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class BorderComponent(Component):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node

    def draw(self, ax: plt.Axes):
        x = self.node[0]
        y = self.node[1]
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor='lightgrey', linewidth=2))

    def add_constraints(self, router: "Router"):
        router.add_null_node_constraints(self.node)
