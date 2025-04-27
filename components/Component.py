import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from constants import Node

if TYPE_CHECKING:
    from Router import Router

class Component:
    def __init__(self):
        self.io_color = {}
        pass

    def draw(self, ax: plt.Axes):
        pass

    def add_constraints(self, router: "Router"):
        pass

    def get_io(self, ith: int = 0):
        pass

    def register_color(self, node: Node, color: str):
        self.io_color[node] = color
    
    def get_nodes(self):
        pass