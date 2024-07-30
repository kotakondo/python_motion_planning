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

from .graph_search3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Node3D

class AStar3D(GraphSearcher3D):
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
    def __init__(self, start: tuple, goal: tuple, env: Env3D, bound: float, heuristic_type: str = "euclidean", weight: float = 1.0) -> None:
        super().__init__(start, goal, env, heuristic_type, weight)

        self.jump_bound = bound
        self.weight = weight

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
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractNodePath(node)
                return cost, path, list(CLOSED.values())

            for node_n in self.getNeighbor(node):                
                # exists in CLOSED list
                if node_n.current in CLOSED:
                    continue
                
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

    def extractNodePath(self, node: Node3D) -> tuple:
        """
        Extract the path based on the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            cost (float): the cost of planned path
            path (list): the planning path
        """
        cost = node.g
        path = [node]
        parent = node.parent
        while parent:
            path.append(parent)
            parent = parent.parent
        return cost, path
    
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
    
    def extractAllNodes(self, closed_list: dict) -> list:
        """
        Extract all nodes from the CLOSED list.

        Parameters:
            closed_list (dict): CLOSED list

        Returns:
            all_nodes (list): all nodes in the CLOSED list
        """
        all_nodes = []
        for node in closed_list.values():
            all_nodes.append(node.current)
        return all_nodes

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
        
        # save the image
        # self.save_final_path_figure(path, cost, f"{self}_bound_{self.jump_bound}_weight_{self.weight}.png")
        self.plot.animation(path, str(self), cost)

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

    def compute_total_time(self, node1: Node3D, node2: Node3D) -> tuple:
        """
        Compute the distance and wait time.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            dist (float): distance between node1 and node2
            wait_time (float): wait time
        """

        total_time = math.sqrt((node2.current[0] - node1.current[0]) ** 2 + (node2.current[1] - node1.current[1]) ** 2) + node2.wait_time

        return total_time

    def save_final_path_figure(self, path: list, cost: float, fig_name: str) -> None:
        """
        Save the final path as a figure.
        """

        # get interpolated path
        interpolated_path = self.env.interpolate_path(path)

        # generate random colors
        colors = [np.random.rand(3,) for _ in range(100)]

        # get interval path
        interval_paths = self.get_interval_paths(interpolated_path, cost, colors)

        # debug
        self.plot.dynamic_animation(interval_paths, str(self), cost, [], colors, fig_name=fig_name)

    def get_interval_paths(self, interpolated_path: list, cost: float = 0, colors: list = []) -> list:

        interval_paths = []
        dist = 0
        while len(interpolated_path) > 1:

            # get the interval index
            for i in range(len(interpolated_path) -1 ):

                # get the next node
                current_node = interpolated_path[i]
                next_node = interpolated_path[i + 1]

                # check if the next node is in the same interval
                dist = dist + math.sqrt((next_node[0] - current_node[0]) ** 2 + (next_node[1] - current_node[1]) ** 2)

                if dist > self.jump_bound:
                    
                    # get this interval path
                    interval_paths.append(interpolated_path[:i+1])

                    # remove the interval path from the interpolated path
                    interpolated_path = interpolated_path[i+1:]

                    dist = dist - self.jump_bound

                    interpolated_path = [] if next_node == interpolated_path[-1] else interpolated_path

                    break

                if next_node == interpolated_path[-1]:

                    interval_paths.append(interpolated_path)
                    interpolated_path = []

                    break

            # connect interval paths for visualization
            # self.connect_intervals_for_visualization(interval_paths)

            # debug (visualize each interval)
            # self.plot.dynamic_animation(interval_paths, str(self), cost, [], colors)

        # connect interval paths for visualization
        self.connect_intervals_for_visualization(interval_paths)

        return interval_paths
    
    def connect_intervals_for_visualization(self, interval_paths: list) -> list:
        """
        Connect intervals for visualization.

        Parameters:
            interval_paths (list): interval paths

        """

        for idx in range(1, len(interval_paths)):
            interval_paths[idx].insert(0, interval_paths[idx - 1][-1])
    
    def benchmarking_run(self, num_runs: int, save_fig: bool) -> tuple:
        """
        Benchmarking running.
        """

        # initialize lists
        cost_list = np.zeros(num_runs)
        time_list = np.zeros(num_runs)
        success_list = np.zeros(num_runs)

        # run the planner
        for i in range(num_runs):

            # run the planner
            start_time = time.time()
            cost, _, path, _ = self.plan()
            end_time = time.time()

            # update the lists
            cost_list[i] = cost
            time_list[i] = end_time - start_time
            success_list[i] = 1 if path[0] == self.goal.current else 0

            # save the image
            self.save_final_path_figure(path, cost, f"{self}_bound_{self.jump_bound}_weight_{self.weight}.png") if save_fig else None
        
        # take the average
        avg_cost = np.mean(cost_list)
        avg_time = np.mean(time_list)
        success_rate = np.mean(success_list)
        
        return avg_cost, avg_time, success_rate
    
    def update_env(self, node: Node3D) -> None:

        # get the interval indecies we need to consider
        inference_start = time.time()
        current_index, dist_so_far_interval = self.get_current_index(node)

        # create interval_indecies
        interval_indecies = []
        interval_indecies.append(current_index) if dist_so_far_interval + 1.0 < self.jump_bound else None # if any next step (when node.g + 1.0 (not 1.41)) will jump to the next interval, then we don't need to consider the current interval
        interval_indecies.append(current_index + 1) # when jumped, the agent could land in the next interval
        interval_indecies.append(current_index + 2) # when jumped, the agent could land in the next interval

        # update the environment
        self.env.update(interval_indecies)
        self.obstacles = self.env.obstacles
        self.inference_time += time.time() - inference_start