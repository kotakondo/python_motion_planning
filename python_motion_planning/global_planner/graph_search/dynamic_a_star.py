"""
@file: a_star.py
@breif: A* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq
import math
from math import sqrt
import numpy as np
import time

from .a_star import AStar
from python_motion_planning.utils import Env, Node
import matplotlib.pyplot as plt

class DynamicAStar(AStar):
    """
    Class for Dynamic A* motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.AStar((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] A Formal Basis for the heuristic Determination of Minimum Cost Paths
    """
    def __init__(self, start: tuple, goal: tuple, env: Env, bound: float, weight: float, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

        # set class variable
        self.jump_bound = bound
        self.weight = weight

        # change start and goal's weight
        self.start.weight = self.weight
        self.goal.weight = self.weight

        # change motions' weight
        self.motions = [Node((-1, 0), None, 1, None, self.weight), Node((-1, 1),  None, sqrt(2), None, self.weight),
                        Node((0, 1),  None, 1, None, self.weight), Node((1, 1),   None, sqrt(2), None, self.weight),
                        Node((1, 0),  None, 1, None, self.weight), Node((1, -1),  None, sqrt(2), None, self.weight),
                        Node((0, -1), None, 1, None, self.weight), Node((-1, -1), None, sqrt(2), None, self.weight)]

    def __str__(self) -> str:
        return "Dynamic A*"

    def plan(self) -> tuple:
        """
        Dynamic A* motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        # OPEN list (priority queue) and CLOSED list (hash table)
        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()

        while OPEN:

            # pop the node with the smallest f value
            node = heapq.heappop(OPEN)

            # get the interval indecies we need to consider
            current_index, total_time_so_far_interval = self.get_current_index(node)

            # create interval_indecies
            interval_indecies = []

            # a bit conservative here but if the agent takes the next step diagonally, it could add 1.0 to the cost, and if that exceeds the jump_bound, the agent could land in the next interval
            interval_indecies.append(current_index) if (total_time_so_far_interval + 1.0) < self.jump_bound else interval_indecies.append(current_index + 1)

            # update the environment
            self.env.update(interval_indecies)
            self.obstacles = self.env.obstacles

            # exists in CLOSED list
            if node.current in CLOSED:                
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            # get neighbors
            for node_n in self.getNeighbor(node):    

                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue
                
                # update node_n
                node_n.parent = node
                node_n.h = self.h(node_n, self.goal)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[node.current] = node
        return [], [], []
    
    def getNeighbor(self, node: Node) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (Node): current node

        Returns:
            neighbors (list): neighbors of current node
        """

        # we can only wait if there is an obstacle around the current node
        return [node + motion for motion in self.motions if not self.isCollisionModified(node, node + motion)]

    def run(self):
        """
        Running both planning and animation.
        """

        start_time = time.time()
        cost, path, expand = self.plan()
        end_time = time.time()

        print("cost: ", cost)
        print("time: ", end_time - start_time)

        # save the image
        self.save_final_path_figure(path, cost, f"{self}_bound_{self.jump_bound}_weight_{self.weight}.png")