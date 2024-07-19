"""
@file: a_star.py
@breif: A* motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq
import math
import numpy as np

from .graph_search import GraphSearcher
from python_motion_planning.utils import Env, Node


class AStar(GraphSearcher):
    """
    Class for A* motion planning.

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
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

        # only for BJPS
        # define the maximum search depth (bound)
        self.jump_bound = 5.0 # hard code for now

    def __str__(self) -> str:
        return "A*"

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
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue
                
                node_n.parent = node.current
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
        return [node + motion for motion in self.motions
                if not self.isCollision(node, node + motion)]

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
            node_parent = closed_list[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def run(self):
        """
        Running both planning and animation.
        """
        cost, path, expand = self.plan()
        print("cost: ", cost)
        print("path: ", path)

        # dynamic animation

        # flip the path list
        path = path[::-1]

        # get interpolated path 
        # currently the path only contrains the nodes; we need to interpolate the path
        interpolated_path = []
        interpolated_path.append(path[0])
        for current_node_idx, current_node in enumerate(path[:-1]):

            next_node = path[current_node_idx + 1]

            # determine the direction of the path
            direction = (next_node[0] - current_node[0], next_node[1] - current_node[1])
            
            # check if the path is horizontal, vertical, or diagonal
            motion = (direction[0] // abs(direction[0]) if direction[0] != 0 else 0, direction[1] // abs(direction[1]) if direction[1] != 0 else 0)

            # initialize interpolated node
            interpolated_node = current_node

            # add interpolated nodes until the next node
            while interpolated_node != next_node:
                interpolated_node = (interpolated_node[0] + motion[0], interpolated_node[1] + motion[1])
                interpolated_path.append(interpolated_node)
        
        print("interpolated_path: ", interpolated_path)

        # visualization colors
        # generate random colors
        colors = [np.random.rand(3,) for _ in range(100)]

        # get interval path
        interval_paths = []
        while len(interpolated_path) > 0:

            # get the interval index
            dist = 0
            for i in range(len(interpolated_path) - 1):

                # get the next node
                current_node = interpolated_path[i]
                next_node = interpolated_path[i + 1]

                # check if the next node is in the same interval
                dist = dist + math.sqrt((next_node[0] - current_node[0]) ** 2 + (next_node[1] - current_node[1]) ** 2)
                if dist > self.jump_bound:
                    
                    # get this interval path
                    interval_paths.append(interpolated_path[:i+1])

                    # remove the interval path from the interpolated path
                    interpolated_path = interpolated_path[i:]

                    break

                if next_node == interpolated_path[-1]:
                        
                    # get this interval path
                    interval_paths.append(interpolated_path)

                    # remove the interval path from the interpolated path
                    interpolated_path = []

                    break

            # debug
            self.plot.dynamic_animation(interval_paths, str(self), cost, [], colors)

        print("interval_paths: ", interval_paths)

        # debug
        self.plot.dynamic_animation(interval_paths, str(self), cost, [], colors)

        # static animation
        # self.plot.animation(path, str(self), cost, expand)
