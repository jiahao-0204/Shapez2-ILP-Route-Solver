# Copyright (c) 2025 Jiahao
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
from constants import Node

if TYPE_CHECKING:
    from Router import Router

class Component:
    def __init__(self):
        self.io_color = {}
        pass
    
    def register_color(self, node: Node, color: str):
        self.io_color[node] = color

    def draw(self, ax: plt.Axes):
        # define the drawing logic for the component
        pass

    def add_constraints(self, router: "Router"):
        # define the io constraints for the component
        pass

    def get_io(self, ith: int = 0):
        # get the input and output nodes for the component, for use in router net definition
        pass
    
    def get_nodes(self):
        # get the actual occupied nodes for the component, for use in router border generation
        pass