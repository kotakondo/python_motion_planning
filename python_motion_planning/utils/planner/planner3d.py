"""
@file: planner.py
@breif: Abstract class for planner
@author: Winter
@update: 2023.1.17
"""
import math
from abc import abstractmethod, ABC
from ..environment.env3d import Env3D, Node3D
from ..plot.plot3d import Plot3D

class Planner3D(ABC):
    def __init__(self, start: tuple, goal: tuple, env: Env3D, weight: float = 1.0) -> None:
        # plannig start and goal

        self.start = Node3D(start, None, 0, 0, weight)
        self.goal = Node3D(goal, None, 0, 0, weight)
        # environment
        self.env = env
        # graph handler
        self.plot = Plot3D(start, goal, env)

    def dist(self, node1: Node3D, node2: Node3D) -> float:
        return math.hypot(node2.x - node1.x, node2.y - node1.y, node2.z - node1.z)
    
    @abstractmethod
    def plan(self):
        '''
        Interface for planning.
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Interface for running both plannig and animation.
        '''
        pass