"""
@file: a_star.py
@breif: A* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq
import math
import numpy as np
import time

from .a_star3d import AStar3D
from python_motion_planning.utils import Env3D, Node3D

class DynamicAStar3D(AStar3D):
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
    def __init__(self, start: tuple, goal: tuple, env: Env3D, bound: float, weight: float = 1.0, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, bound, heuristic_type, weight)

        self.jump_bound = bound
        self.weight = weight

        # debug
        self.inference_time = 0.0

        # allow non-move motion (add 1 to g value)
        # self.motions.append(Node3D((0, 0, 0), None, 0.3, None, self.weight, wait_time=1.0))

    def __str__(self) -> str:
        return "A* 3D"

    def plan(self) -> tuple:
        """
        A* motion plan function.

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
            node = heapq.heappop(OPEN)

            # exists in CLOSED list
            if (node.current, node.wait_time) in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[(node.current, node.wait_time)] = node
                cost, node_path = self.extractNodePath(node)
                print(f"Dynamic A* Inference Time: {self.inference_time}")
                return cost, node_path[::-1], list(CLOSED.values())

            # get the interval indecies we need to consider
            inference_start = time.time()
            current_index, total_time_so_far_interval = self.get_current_index(node)

            # create interval_indecies
            interval_indecies = []

            # a bit conservative here but if the agent takes the next step diagonally, it could add 1.0 to the cost, and if that exceeds the jump_bound, the agent could land in the next interval
            interval_indecies.append(current_index) if (total_time_so_far_interval + 1.0) < self.jump_bound else interval_indecies.append(current_index + 1)

            # update the environment
            self.env.update(interval_indecies)
            self.obstacles = self.env.obstacles
            self.inference_time += time.time() - inference_start

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED list
                if (node_n.current, node_n.wait_time) in CLOSED:
                    continue
                
                node_n.parent = node
                node_n.h = self.h(node_n, self.goal)

                # goal found
                if node_n == self.goal:
                    heapq.heappush(OPEN, node_n)
                    break
                
                # update OPEN list
                heapq.heappush(OPEN, node_n)

            CLOSED[(node.current, node.wait_time)] = node
        return [], [], []

    def getNeighbor(self, node: Node3D) -> list:
        """
        Find neighbors of node.

        Parameters:
            node (Node): current node

        Returns:
            neighbors (list): neighbors of current node
        """

        return [node + motion for motion in self.motions
                if not self.isCollisionModified(node, node + motion)]

    def extractPath(self, closed_list: dict) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planned path
            path (list): the planning path
        """
        cost = 0
        node = closed_list[self.goal.current]
        path = [node.current]
        while node != self.start:
            node_parent = closed_list[node.parent.current]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def get_current_index(self, node: Node3D) -> tuple:
        """
        we need to track which interval the agent is in by considering the path it has traversed 
        example 
        interval 0: [0, 4.2]
        interval 1: [4.2, 9.3]
        interval 2: if we do g // 5.0, which is 9.3 // 5.0 = 1. But it's actually in interval 2 not 1 

        Parameters:
            node (Node): current node

        Returns:
            current_index (int): current index
        """

        # get all the parents of the node

        current_node_parent = node.parent
        path = []
        path.append(node)

        while current_node_parent:
            path.append(current_node_parent)
            current_node_parent = current_node_parent.parent

        current_node_index, total_time_so_far_interval = self.get_current_index_from_path(path)

        return current_node_index, total_time_so_far_interval
    
    def get_current_index_from_path(self, path: list) -> tuple:
        """
        Get the current index of the node.

        Parameters:
            path (list): path

        Returns:
            current_index (int): current index
        """

        # flip the path list (so that it will start from the start node)
        path = path[::-1]

        # get the correct index
        current_node_index = 0
        total_time = 0
        while len(path) > 1:

            # get the interval index
            for i in range(len(path) - 1):

                # get the next node
                current_node = path[i]
                next_node = path[i + 1]

                # check if the next node is in the same interval
                total_time = total_time + self.compute_total_time(current_node, next_node)

                if total_time > self.jump_bound:
                    
                    # get this interval path
                    current_node_index = current_node_index + 1

                    # remove the interval path from the interpolated path
                    path = path[i+1:]

                    total_time = total_time - self.jump_bound

                    path = [] if next_node == path[-1] else path

                    break

                if next_node == path[-1]:

                    path = []

                    break

        return current_node_index, total_time