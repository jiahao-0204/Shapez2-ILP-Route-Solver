from components.Component import Component
from constants import Node
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class BorderComponent(Component):
    def __init__(self, node: Node):
        super().__init__()

        # node
        self.node = node

        # x and y
        self.x = node[0]
        self.y = node[1]

    def draw(self, ax: plt.Axes):
        ax.add_patch(plt.Rectangle((self.x, self.y), 1, 1, facecolor='lightgrey', linewidth=2))

    def add_constraints(self, router: "Router"):
        router.add_null_node_constraints(self.node)
    
    def get_nodes(self):
        return [self.node]