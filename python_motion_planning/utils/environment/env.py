"""
@file: env.py
@breif: 2-dimension environment
@author: Winter
@update: 2023.1.13
"""
from math import sqrt
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

from .node import Node

class Env(ABC):
    """
    Class for building 2-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env
        >>> env = Env(30, 40)
    """
    def __init__(self, x_range: int, y_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j) for i in range(self.x_range) for j in range(self.y_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid(Env):
    """
    Class for discrete 2-d grid map.
    """
    def __init__(self, x_range: int, y_range: int, bound: int) -> None:
        super().__init__(x_range, y_range)
        # allowed motions
        self.motions = [Node((-1, 0), None, 1, None), Node((-1, 1),  None, sqrt(2), None),
                        Node((0, 1),  None, 1, None), Node((1, 1),   None, sqrt(2), None),
                        Node((1, 0),  None, 1, None), Node((1, -1),  None, sqrt(2), None),
                        Node((0, -1), None, 1, None), Node((-1, -1), None, sqrt(2), None)]
        
        # obstacles
        self.static_obstacles = None
        self.dynamic_obstacles = None
        self.dynamic_obstacles_objects = []
        self.obstacles = None
        self.obstacles_tree = None
        
        # interval parameters
        self.bound = bound

        # initialize grid map
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y = self.x_range, self.y_range
        static_obstacles = set()
        dynamic_obstacles = set()

        # boundary of environment
        for i in range(x):
            static_obstacles.add((i, 0))
            static_obstacles.add((i, y - 1))
        for i in range(y):
            static_obstacles.add((0, i))
            static_obstacles.add((x - 1, i))

        # initialize static obstacles
        for i in range(5, 21):
            static_obstacles.add((i, 15))
        for i in range(15):
            static_obstacles.add((20, i))
        for i in range(15, 30):
            static_obstacles.add((30, i))
        for i in range(16):
            static_obstacles.add((40, i))

        # initialize dynamic obstacles
        self.create_dynamic_obstacles()

        # add dynamic obstacles to obstacles
        for dyn_obst_object in self.dynamic_obstacles_objects:
            dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([0])) # 0 because the initial index is 0

        # update obstacles
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.obstacles = static_obstacles.union(dynamic_obstacles)
        self.obstacles_tree = cKDTree(np.array(list(self.obstacles)))

    def create_dynamic_obstacles(self) -> None:
        """
        Create dynamic obstacles.
        """

        # dynamic obstacle 1
        path1 = []
        path1.append((25, 1))  # start point
        path1.append((25, 25)) # middle point 1
        path1.append((5, 25))  # middle point 2
        path1.append((15, 25)) # end point
        flipped_path1 = self.interpolate_path(path1)
        path1 = flipped_path1[::-1] # need to flip the path to make it compatible with the current implementation
        path1 = path1 + flipped_path1
        vel1 = 1
        self.dynamic_obstacles_objects.append(Grid.DynamicObstacles(path1, vel1, self.bound))

        # dynamic obstacle 2
        path2 = []
        path2.append((35, 1))  # start point
        path2.append((35, 25)) # middle point 1
        path2.append((45, 25)) # middle point 2
        path2.append((45, 10)) # end point
        flipped_path2 = self.interpolate_path(path2)
        path2 = flipped_path2[::-1] # need to flip the path to make it compatible with the current implementation
        path2 = path2 + flipped_path2
        vel2 = 2
        self.dynamic_obstacles_objects.append(Grid.DynamicObstacles(path2, vel2, self.bound))

    def interpolate_path(self, path: list) -> list:
        """
        Interpolrate path.

        Parameters:
            path (list): path

        Returns:
            interpolated_path (list): interpolated path
        """
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

        return interpolated_path

    # update the dynamic obstacles
    def update(self, interval_indecies: list) -> None:
        self.update_dynamic_obstacles(interval_indecies)
        self.obstacles_tree = cKDTree(np.array(list(self.obstacles)))

    def update_dynamic_obstacles(self, interval_indecies: list) -> None:
        """
        remove the dynamic obstacles
        """

        # clean up the dynamic obstacles
        self.dynamic_obstacles.clear()
        self.obstacles.clear()
        
        # add dynamic obstacles according to the interval_indecies
        dynamic_obstacles = set()
        for interval_index in interval_indecies:
            for dyn_obst_object in self.dynamic_obstacles_objects:
                dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([interval_index]))
        
        # update obstacles
        self.obstacles = self.static_obstacles.union(dynamic_obstacles)
        self.dynamic_obstacles = dynamic_obstacles

    def get_obstacles_from_interval_index_for_plot(self, interval_index: int) -> set:
        
        # copy obstacles
        static_obstacles = self.static_obstacles.copy()
        
        # add dynamic obstacles according to the interval_indecies
        dynamic_obstacles = set()
        for dyn_obst_object in self.dynamic_obstacles_objects:
            dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([interval_index]))

        return dynamic_obstacles, static_obstacles

    # nested class for dynamic obstacles
    class DynamicObstacles:
        
        # @param path: list of positions of dynamic obstacles
        # @param vel: velocity of dynamic obstacles = occupied pixels per interval
        def __init__(self, path: list, vel: float, bound: float) -> None:
            self.path = path
            self.vel = vel
            self.bound = bound
        
        # @param interval_index: index of the interval
        # @return: set of dynamic obstacles
        def get_hull_from_interval_indecies(self, interval_indecies: list[int]) -> set:
            
            # initialize obstacles
            obstacles = set()

            # loop through the interval_indecies
            for interval_index in interval_indecies:

                # calculate the initial index of the interval
                interval_initial_index = int(interval_index * (self.vel * self.bound)) # vel [pixels/s] * bound [s/interval] = [pixels/interval]
                
                # add dynamic obstacles according to the interval_index
                for i in range(interval_initial_index, int(interval_initial_index + self.vel * self.bound)):

                    # if i is out of the path, return the first index + however many indexes are left
                    i = i % len(self.path)

                    # add the obstacle
                    obstacles.add(self.path[i])
            
            return obstacles


class Map(Env):
    """
    Class for continuous 2-d map.
    """
    def __init__(self, x_range: int, y_range: int) -> None:
        super().__init__(x_range, y_range)
        self.boundary = None
        self.obs_circ = None
        self.obs_rect = None
        self.init()

    def init(self):
        """
        Initialize map.
        """
        x, y = self.x_range, self.y_range

        # boundary of environment
        self.boundary = [
            [0, 0, 1, y],
            [0, y, x, 1],
            [1, 0, x, 1],
            [x, 1, 1, y]
        ]

        # user-defined obstacles
        self.obs_rect = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]

        self.obs_circ = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

    def update(self, boundary, obs_circ, obs_rect):
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect
