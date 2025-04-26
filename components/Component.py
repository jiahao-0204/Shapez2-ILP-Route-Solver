import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Router import Router

class Component:
    def __init__(self):
        pass

    def draw(self, ax: plt.Axes):
        pass

    def add_constraints(self, router: "Router"):
        pass