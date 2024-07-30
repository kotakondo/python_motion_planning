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

from .node3d import Node3D

class Env3D(ABC):
    """
    Class for building 3-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        z_range (int): z-axis range of environmet
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env
        >>> env = Env(30, 40)
    """
    def __init__(self, x_range: int, y_range: int, z_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.z_range = z_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j, k) for i in range(self.x_range) for j in range(self.y_range) for k in range(self.z_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid3D(Env3D):
    """
    Class for discrete 3-d grid map.
    """
    def __init__(self, x_range: int, y_range: int, z_range: int, bound: int, weight: float = 1.0, is_static: bool = False) -> None:
        super().__init__(x_range, y_range, z_range)
        self.weight = weight

        # allowed motions
        self.motions = [Node3D((-1, -1, -1), None, sqrt(3), None, self.weight), Node3D((-1, -1, 0), None, sqrt(2), None, self.weight), Node3D((-1, -1, 1), None, sqrt(3), None, self.weight),
                        Node3D((-1, 0, -1), None, sqrt(2), None, self.weight),  Node3D((-1, 0, 0), None, 1, None, self.weight),        Node3D((-1, 0, 1), None, sqrt(2), None, self.weight),
                        Node3D((-1, 1, -1), None, sqrt(3), None, self.weight),  Node3D((-1, 1, 0), None, sqrt(2), None, self.weight),  Node3D((-1, 1, 1), None, sqrt(3), None, self.weight),
                        Node3D((0, -1, -1), None, sqrt(2), None, self.weight),  Node3D((0, -1, 0), None, 1, None, self.weight),        Node3D((0, -1, 1), None, sqrt(2), None, self.weight),
                        Node3D((0, 0, -1), None, 1, None, self.weight),                                                                Node3D((0, 1, 1), None, sqrt(2), None, self.weight),
                        Node3D((0, 1, -1), None, sqrt(2), None, self.weight),   Node3D((0, 1, 0), None, 1, None, self.weight),         Node3D((0, 1, 1), None, sqrt(2), None, self.weight),
                        Node3D((1, -1, -1), None, sqrt(3), None, self.weight),  Node3D((1, -1, 0), None, sqrt(2), None, self.weight),  Node3D((1, -1, 1), None, sqrt(3), None, self.weight),
                        Node3D((1, 0, -1), None, sqrt(2), None, self.weight),   Node3D((1, 0, 0), None, 1, None, self.weight),         Node3D((1, 0, 1), None, sqrt(2), None, self.weight),
                        Node3D((1, 1, -1), None, sqrt(3), None, self.weight),   Node3D((1, 1, 0), None, sqrt(2), None, self.weight),   Node3D((1, 1, 1), None, sqrt(3), None, self.weight)]
        
        # obstacles
        self.static_obstacles = None
        self.dynamic_obstacles = None
        self.dynamic_obstacles_objects = []
        self.obstacles = None
        self.obstacles_tree = None
        self.is_static = is_static
        
        # interval parameters
        self.bound = bound

        # initialize grid map
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y, z = self.x_range, self.y_range, self.z_range
        static_obstacles = set()
        dynamic_obstacles = set()

        # boundary of environment
        for j in range(z):
            for i in range(x):
                static_obstacles.add((i, 0, j))
                static_obstacles.add((i, y - 1, j))
            for i in range(y):
                static_obstacles.add((0, i, j))
                static_obstacles.add((x - 1, i, j))
        
        # ceiling and floor
        for i in range(x):
            for j in range(y):
                static_obstacles.add((i, j, -1))
                static_obstacles.add((i, j, z))

        # initialize static obstacles
        for j in range(z):
        # for j in range(3):
            for i in range(5, 21):
                static_obstacles.add((i, 15, j))
            for i in range(15):
                static_obstacles.add((20, i, j))
            for i in range(15, 30):
                static_obstacles.add((30, i, j))
            for i in range(16):
                static_obstacles.add((40, i, j))

        # initialize dynamic obstacles
        self.create_dynamic_obstacles()

        # depending on is_static, add dynamic obstacles to dynamic_obstacles
        if not self.is_static:

            # add dynamic obstacles to dynamic_obstacles
            for dyn_obst_object in self.dynamic_obstacles_objects:
                dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([0])) # 0 because the initial index is 0

        else:
            
            # add dynamic obstacles to static obstacles
            for dyn_obst_object in self.dynamic_obstacles_objects:
                static_obstacles.update(dyn_obst_object.get_path_as_obstacles())

        # update obstacles
        self.static_obstacles = static_obstacles
        self.dynamic_obstacles = dynamic_obstacles if not self.is_static else set()
        self.obstacles = static_obstacles.union(dynamic_obstacles) if not self.is_static else static_obstacles
        self.obstacles_tree = cKDTree(np.array(list(self.obstacles)))

    def create_dynamic_obstacles(self) -> None:
        """
        Create dynamic obstacles.
        """

        # dynamic obstacle 1
        # for j in range(1, self.z_range-1):
        for j in range(self.z_range):
        # for j in range(3):
            path1 = []
            path1.append((25, 1, j))  # start point
            path1.append((25, 25, j)) # middle point 1
            path1.append((5, 25, j))  # middle point 2
            path1.append((15, 25, j)) # end point
            flipped_path1 = self.interpolate_path(path1)
            path1 = flipped_path1[::-1] # need to flip the path to make it compatible with the current implementation
            path1 = path1 + flipped_path1
            vel1 = 3
            self.dynamic_obstacles_objects.append(Grid3D.DynamicObstacles(path1, vel1, self.bound))

        # dynamic obstacle 2
        # for j in range(1, self.z_range-1):
        for j in range(self.z_range):
        # for j in range(3):
            path2 = []
            path2.append((35, 1, j))  # start point
            path2.append((35, 25, j)) # middle point 1
            path2.append((45, 25, j)) # middle point 2
            path2.append((45, 10, j)) # end point
            flipped_path2 = self.interpolate_path(path2)
            path2 = flipped_path2[::-1] # need to flip the path to make it compatible with the current implementation
            path2 = path2 + flipped_path2
            vel2 = 2
            self.dynamic_obstacles_objects.append(Grid3D.DynamicObstacles(path2, vel2, self.bound))

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
            direction = (next_node[0] - current_node[0], next_node[1] - current_node[1], next_node[2] - current_node[2])
            
            # check if the path is horizontal, vertical, or diagonal
            motion = (direction[0] // abs(direction[0]) if direction[0] != 0 else 0, direction[1] // abs(direction[1]) if direction[1] != 0 else 0, direction[2] // abs(direction[2]) if direction[2] != 0 else 0)

            # initialize interpolated node
            interpolated_node = current_node

            # add interpolated nodes until the next node
            while interpolated_node != next_node:
                interpolated_node = (interpolated_node[0] + motion[0], interpolated_node[1] + motion[1], interpolated_node[2] + motion[2])
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

    def get_dynamic_obstacles(self, interval_index: int) -> None:
        """
        get the dynamic obstacles
        """

        # add dynamic obstacles according to the interval_indecies
        dynamic_obstacles = set()
        for dyn_obst_object in self.dynamic_obstacles_objects:
            dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([interval_index]))
        
        return dynamic_obstacles
    
    def get_obstacles_from_interval_index_for_plot(self, interval_index: int) -> set:
        
        # copy obstacles
        static_obstacles = self.static_obstacles.copy()
        
        # add dynamic obstacles according to the interval_indecies
        dynamic_obstacles = set()
        for dyn_obst_object in self.dynamic_obstacles_objects:
            dynamic_obstacles.update(dyn_obst_object.get_hull_from_interval_indecies([interval_index]))


        # filter to get rid of walls and ceiling and floor
        static_obstacles = self.filter_for_plot(static_obstacles)
        dynamic_obstacles = self.filter_for_plot(dynamic_obstacles)

        return dynamic_obstacles, static_obstacles
    
        # for plot, we want a dense map so devide the z-axis by 10
        # dense_static_obstacles = set()
        # for static_obstacle in static_obstacles:
        #     x, y, z = static_obstacle
        #     dense_static_obstacles.add((x, y, z / 10.0))

        # dense_dynamic_obstacles = set()
        # for dynamic_obstacle in dynamic_obstacles:
        #     x, y, z = dynamic_obstacle
        #     dense_dynamic_obstacles.add((x, y, z / 10.0))

        # return dense_dynamic_obstacles, dense_static_obstacles

    def filter_for_plot(self, obstacles: set) -> tuple:
        '''
        Filter walls from obstacles.

        Parameters
        ----------
        obs_x: X coordinates of obstacles
        obs_y: Y coordinates of obstacles
        obs_z: Z coordinates of obstacles

        Returns
        -------
        filtered_obs: filtered obstacles
        '''

        obs_x = [x[0] for x in obstacles]
        obs_y = [x[1] for x in obstacles]
        obs_z = [x[2] for x in obstacles]

        filtered_obs_x, filtered_obs_y, filtered_obs_z = [], [], []
        for i in range(len(obs_x)):
            if not (obs_x[i] == 0 or obs_x[i] == self.x_range -1 or \
                obs_y[i] == 0 or obs_y[i] == self.y_range -1 or \
                obs_z[i] == -1 or obs_z[i] == self.z_range):
                filtered_obs_x.append(obs_x[i])
                filtered_obs_y.append(obs_y[i])
                filtered_obs_z.append(obs_z[i])

        # reconstruct obstacles
        filtered_obs = set()
        for i in range(len(filtered_obs_x)):
            filtered_obs.add((filtered_obs_x[i], filtered_obs_y[i], filtered_obs_z[i]))

        return filtered_obs

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
        
        def get_path_as_obstacles(self) -> set:
            return set(self.path)
