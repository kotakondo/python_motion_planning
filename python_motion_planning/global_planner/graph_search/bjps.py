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
import tqdm

from .a_star import AStar
from python_motion_planning.utils import Env, Node


class BJPS(AStar):
    """
    Class for Bounded JPS motion planning.

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
    def __init__(self, start: tuple, goal: tuple, env: Env, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
    
    def __str__(self) -> str:
        return "Bounded Jump Point Search(BJPS)"

    def plan(self) -> tuple:
        """
        BJPS motion plan function.

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

            # get the interval indecies we need to consider
            current_index = int(node.g // self.jump_bound)

            # create interval_indecies
            interval_indecies = []
            interval_indecies.append(current_index)
            interval_indecies.append(current_index + 1) # when jumped, the agent could land in the next interval
            interval_indecies.append(current_index + 2) # when jumped, the agent could land in the next interval

            # update the environment
            self.env.update(interval_indecies)

            # debug visualization
            # self.plot.animation([], str(self), 0, [])

            # exists in CLOSED list
            if node.current in CLOSED:
                continue

            # goal found
            if node == self.goal:
                CLOSED[node.current] = node
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            jp_list = []
            for motion in self.motions:
                jp = self.jump(node, motion, node)
                # exists and not in CLOSED list
                if jp and jp.current not in CLOSED:
                    jp.parent = node.current
                    jp.h = self.h(jp, self.goal)
                    jp_list.append(jp)
                
            # print the jump point added in this iteration
            # debug
            # print("parent: ", node.current)
            # for idx, tmp in enumerate(jp_list):
            #     print("jp: ", tmp.current)
            #     print("dist: ", self.dist(tmp, node))

            for jp in jp_list:
                # update OPEN list
                heapq.heappush(OPEN, jp)

                # goal found
                if jp == self.goal:
                    break

            CLOSED[node.current] = node
        return [], [], []

    def jump(self, node: Node, motion: Node, jump_point: Node) -> Node:
        """
        Jumping search recursively.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes

        Returns:
            jump_point (Node): jump point or None if searching fails
        """
        # explore a new node
        new_node = node + motion
        new_node.parent = node.current
        new_node.h = self.h(new_node, self.goal)


        # hit the obstacle
        if new_node.current in self.obstacles:
            return None

        # goal found
        if new_node == self.goal:
            return new_node

        # diagonal
        if motion.x and motion.y:
            # if exists jump point at horizontal or vertical
            x_dir = Node((motion.x, 0), None, 1, None, self.weight)
            y_dir = Node((0, motion.y), None, 1, None, self.weight)
            if self.jump(new_node, x_dir, jump_point) or self.jump(new_node, y_dir, jump_point):
                return new_node
        
        # check if the new node is within the boundary
        dist = self.dist(new_node, jump_point)
        direction_dist = self.dist(motion, Node((0, 0), None, 1, None, self.weight))
        if dist + direction_dist > self.jump_bound:
            return new_node 
            
        # if exists forced neighbor
        if self.detectForceNeighbor(new_node, motion):
            return new_node
        else:
            return self.jump(new_node, motion, jump_point)

    def detectForceNeighbor(self, node, motion):
        """
        Detect forced neighbor of node.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes

        Returns:
            flag (bool): True if current node has forced neighbor else Flase
        """
        x, y = node.current
        x_dir, y_dir = motion.current

        # horizontal
        if x_dir and not y_dir:
            if (x, y + 1) in self.obstacles and \
                (x + x_dir, y + 1) not in self.obstacles:
                return True
            if (x, y - 1) in self.obstacles and \
                (x + x_dir, y - 1) not in self.obstacles:
                return True
        
        # vertical
        if not x_dir and y_dir:
            if (x + 1, y) in self.obstacles and \
                (x + 1, y + y_dir) not in self.obstacles:
                return True
            if (x - 1, y) in self.obstacles and \
                (x - 1, y + y_dir) not in self.obstacles:
                return True
        
        # diagonal
        if x_dir and y_dir:
            if (x - x_dir, y) in self.obstacles and \
                (x - x_dir, y + y_dir) not in self.obstacles:
                return True
            if (x, y - y_dir) in self.obstacles and \
                (x + x_dir, y - y_dir) not in self.obstacles:
                return True
        
        return False

    def benchmarking_run(self, num_runs: int, bound: float, weight: float) -> tuple:
        """
        Benchmarking running.
        """

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

        # initialize lists
        cost_list = np.zeros(num_runs)
        time_list = np.zeros(num_runs)
        success_list = np.zeros(num_runs)

        # run the planner
        for i in tqdm.tqdm(range(num_runs)):

            # run the planner
            start_time = time.time()
            cost, path, _ = self.plan()
            end_time = time.time()

            # update the lists
            cost_list[i] = cost
            time_list[i] = end_time - start_time
            success_list[i] = 1 if path[0] == self.goal.current else 0
        
        # take the average
        avg_cost = np.mean(cost_list)
        avg_time = np.mean(time_list)
        success_rate = np.mean(success_list)
        
        return avg_cost, avg_time, success_rate