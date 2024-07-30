"""
@file: jps.py
@breif: Jump Point Search motion planning
@author: Yang Haodong, Wu Maojia
@update: 2024.6.23
"""
import heapq
import numpy as np
from math import sqrt
import time
import matplotlib.pyplot as plt

from .a_star3d import AStar3D
from .jps3d import JPS3D
from .bjps3d import BJPS3D
from .dynamic_a_star3d import DynamicAStar3D
from python_motion_planning.utils import Env3D, Node3D

class DJPS3DPP(AStar3D):
    """
    Class for Dynamic JPS++ motion planning.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Env): environment
        heuristic_type (str): heuristic function type

    Examples:
        >>> import python_motion_planning as pmp
        >>> planner = pmp.JPS((5, 5), (45, 25), pmp.Grid(51, 31))
        >>> cost, path, expand = planner.plan()     # planning results only
        >>> planner.plot.animation(path, str(planner), cost, expand)  # animation
        >>> planner.run()       # run both planning and animation

    References:
        [1] Online Graph Pruning for Pathfinding On Grid Maps
    """
    def __init__(self, start: tuple, goal: tuple, env: Env3D, bound: float, weight: float, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, bound, heuristic_type, weight)

        # initialize static jps 3d planner
        self.static_jps3d_planner = JPS3D(start, goal, env, bound, weight, heuristic_type)

        # initialize dynamic jps 3d planner
        self.dynamic_jps3d_planner = BJPS3D(start, goal, env, bound, weight, heuristic_type)

        # initialize dynamic a star 3d planner
        self.dynamic_a_star3d_planner = DynamicAStar3D(start, goal, env, bound, weight, heuristic_type)


        # initialize parameters
        self.is_dynamic_collision_detected = False

    def __str__(self) -> str:
        return "Bounded Jump Point Search(BJPS) 3D"

    def plan(self) -> tuple:
        """
        Dynaimc JPS++ motion plan function.

        Returns:
            cost (float): path cost
            path (list): planning path
            expand (list): all nodes that planner has searched
        """
        
        # first run static JPS 3D without considering dynamic obstacles
        cost, node_path, _ = self.static_jps3d_planner.plan()

        # go thru the path and check if there's collisions with dynamic obstacles
        node_idx = 1 # skip the start node
        while node_idx < len(node_path) -1: # skip the goal node

            start_node, goal_node, start_node_idx, goal_node_idx = self.get_detour_start_and_goal_nodes(node_path, node_idx)

            # if there's collision with dynamic obstacles
            if self.is_dynamic_collision_detected:
            
                print("collision detected")

                # set start and goal nodes for the detour path
                self.dynamic_a_star3d_planner.start = start_node
                self.dynamic_a_star3d_planner.goal = goal_node

                # include this specific obstacle
                self.dynamic_a_star3d_planner.update_env(start_node) # use node_path[start_node_idx], not start_node, which has 0 cost for plan()

                # replan the path
                detour_cost, detour_node_path, _ = self.dynamic_a_star3d_planner.plan()

                # update the costs of the original path based on the detour path
                node_path[goal_node_idx].g = detour_node_path[-1].g
                for idx in range(goal_node_idx, len(node_path)-1):
                    node_path[idx + 1].g = node_path[idx].g + self.dist(node_path[idx], node_path[idx + 1])

                # update the wait time of the original path based on the detour path
                for idx in range(goal_node_idx+1, len(node_path)):
                    node_path[idx].wait_time = detour_node_path[-1].wait_time

                # insert the detour path into the original path
                node_path = detour_node_path + node_path[goal_node_idx+1:]

                # debug
                # for node in node_path:
                #     print(node.current, node.g)

                # update node_idx
                node_idx = start_node_idx + len(detour_node_path) - 1

            else:
                # if there's no collision with dynamic obstacles
                node_idx += 1

        return node_path[-1].g, node_path, []

    def check_dynamic_obstacle_collision(self, node: Node3D):

        # get the interval indecies we need to consider
        current_index, dist_so_far_interval = self.get_current_index(node)

        # get dynamic obstacles
        dynamic_obstacles = self.env.get_dynamic_obstacles(current_index)

        # check if there's collision with dynamic obstacles
        return node.current in dynamic_obstacles
    
    def get_detour_start_and_goal_nodes(self, node_path: list, node_idx: int) -> tuple:
        """

        Args:
            node_path (list): path of nodes
            node_idx (int): index of the node to check

        Returns:
            start_node (Node3D): start node of the detour path
            goal_node (Node3D): goal node of the detour path
            start_node_idx (int): index of the start node in the original path
            goal_node_idx (int): index of the goal node in the original path
        """

        # find the first node that has collision with dynamic obstacles
        if self.check_dynamic_obstacle_collision(node_path[node_idx]):

            # define the start node
            self.is_dynamic_collision_detected = True
            start_node = Node3D(node_path[node_idx-1].current, node_path[node_idx-1].parent, node_path[node_idx-1].g, node_path[node_idx-1].h, self.weight)
            start_node_idx = node_idx - 1

            # find the goal node (assumption: we can find the path to the goal node)
            for i in range(node_idx+1, len(node_path)):
                if not self.check_dynamic_obstacle_collision(node_path[i]):
                    goal_node = Node3D(node_path[i].current, None, node_path[i].g, node_path[i].h, self.weight)
                    goal_node_idx = i
                    return start_node, goal_node, start_node_idx, goal_node_idx
        
        # if there's no collision with dynamic obstacles
        self.is_dynamic_collision_detected = False
        return None, None, None, None

    def run(self):
        """
        Running both planning and animation.
        """

        start_time = time.time()
        cost, node_path, _ = self.plan()
        end_time = time.time()

        # get the path
        path = [node.current for node in node_path]
        path_with_wait_time = [(node.current, node.wait_time) for node in node_path]

        print("cost: ", cost)
        print("time: ", end_time - start_time)
        print("path_with_wait_time: ", path_with_wait_time)

        self.dynamic_a_star3d_planner.plot.animation(path, str(self), cost, [])