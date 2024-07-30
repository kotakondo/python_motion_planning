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
from python_motion_planning.utils import Env3D, Node3D
from .JPS3DNeib import JPS3DNeib

class BJPS3D(AStar3D):
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
    def __init__(self, start: tuple, goal: tuple, env: Env3D, bound: float, weight: float, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, bound, heuristic_type, weight)

        # set class variable
        self.weight = weight
        self.jump_bound = bound

        # initialize forced neighbor table
        self.jn3d = JPS3DNeib()
        self.jn3d_f1 = self.jn3d.f1

        # debug
        self.inference_time = 0.0

        # allow non-move motion (TODO: hardcoded: add some value to g value)
        self.non_move_motion = Node3D((0, 0, 0), None, 0.3, None, self.weight, wait_time=1.0)

    def __str__(self) -> str:
        return "Bounded Jump Point Search(BJPS) 3D"

    def plan(self) -> tuple:
        """
        JPS motion plan function.

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
                cost, node_path = self.extractNodePath(node)
                print("inference time: ", self.inference_time)
                return cost, node_path[::-1], list(CLOSED.values())

            # update the environment
            self.update_env(node)

            # visualize the environment
            # self.plot.plotCurrentDynamicEnv("tmp") if node.wait_time > 8 else None

            # get successors
            succ_nodes = self.getJpsSucc(node)

            # update OPEN list
            for succ_node in succ_nodes:
                if succ_node.current not in CLOSED:
                    heapq.heappush(OPEN, succ_node)

            CLOSED[node.current] = node
        return [], [], []

    def getJpsSucc(self, node: Node3D):

        succ_nodes = []

        norm1 = abs(node.dx) + abs(node.dy) + abs(node.dz)
        num_neib = self.jn3d.nsz[norm1][0]
        num_fneib = self.jn3d.nsz[norm1][1]
        id = (node.dx + 1) + 3 * (node.dy + 1) + 9 * (node.dz + 1)

        for dev in range(num_neib + num_fneib):
            if dev < num_neib:
                dx = self.jn3d.ns[id][0][dev]
                dy = self.jn3d.ns[id][1][dev]
                dz = self.jn3d.ns[id][2][dev]
                new_node = self.jump(node, Node3D((dx, dy, dz), None, sqrt(dx ** 2 + dy ** 2 + dz ** 2), None, self.weight), node)
                if new_node is None:
                    continue
            else:
                nx = node.current[0] + self.jn3d.f1[id][0][dev - num_neib]
                ny = node.current[1] + self.jn3d.f1[id][1][dev - num_neib]
                nz = node.current[2] + self.jn3d.f1[id][2][dev - num_neib]
                if self.is_occupied(nx, ny, nz):
                    dx = self.jn3d.f2[id][0][dev - num_neib]
                    dy = self.jn3d.f2[id][1][dev - num_neib]
                    dz = self.jn3d.f2[id][2][dev - num_neib]
                    new_node = self.jump(node, Node3D((dx, dy, dz), None, sqrt(dx ** 2 + dy ** 2 + dz ** 2), None, self.weight), node)
                    if new_node is None:
                        continue
                else:
                    continue

            new_node.parent = node
            new_node.h = self.h(new_node, self.goal)
            new_node.dx = dx
            new_node.dy = dy
            new_node.dz = dz
            succ_nodes.append(new_node)

        # add non-move motion
        # new_node = node + self.non_move_motion
        # new_node.parent = node
        # new_node.h = self.h(new_node, self.goal)
        # new_node.dx = 0
        # new_node.dy = 0
        # new_node.dz = 0
        # succ_nodes.append(new_node)
        
        return succ_nodes

    def is_occupied(self, x, y, z):
        """
        Check if the node is occupied.

        Parameters:
            x (int): x coordinate
            y (int): y coordinate
            z (int): z coordinate

        Returns:
            flag (bool): True if the node is occupied else False
        """
        return (x, y, z) in self.obstacles

    def jump(self, node: Node3D, motion: Node3D, original_jp: Node3D) -> Node3D:
        """
        Jumping search recursively in 3D space.

        Parameters:
            node (Node3D): current node
            motion (Node3D): the motion that current node executes

        Returns:
            jump_point (Node3D): jump point or None if searching fails
        """

        if node is None:
            return None
    
        # check if the new node is within the boundary
        dist = self.dist(node, original_jp)
        direction_dist = self.dist(motion, Node3D((0, 0, 0), None, 0, None, self.weight))
        if dist + direction_dist > self.jump_bound:
            return node 

        # Explore a new node
        new_node = node + motion
        new_node.parent = node
        new_node.h = self.h(new_node, self.goal)

        # Hit the obstacle
        if new_node.current in self.obstacles:
            return None

        # Goal found
        if new_node.current == self.goal.current:
            return new_node

        # If exists forced neighbor
        if self.detectForceNeighbor(new_node, motion):
            return new_node
        
        # Jump recursively
        dx = motion.current[0]
        dy = motion.current[1]
        dz = motion.current[2]
 
        id = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
        norm1 = abs(dx) + abs(dy) + abs(dz)
        num_neib = self.jn3d.nsz[norm1][0]

        for k in range(num_neib):
            neib_dx = self.jn3d.ns[id][0][k]
            neib_dy = self.jn3d.ns[id][1][k]
            neib_dz = self.jn3d.ns[id][2][k]

            # print("neib: ", neib_dx, neib_dy, neib_dz) if new_node.current[1] > 25 else None

            new_motion = Node3D((neib_dx, neib_dy, neib_dz), None, sqrt(neib_dx ** 2 + neib_dy ** 2 + neib_dz ** 2), None, self.weight)
            
            if self.jump(new_node, new_motion, original_jp):
                return new_node
        
        # exit() if new_node.current[1] > 25 else None

        return None

    def detectForceNeighbor(self, node, motion):
        """
        Detect forced neighbor of node in 3D space.

        Parameters:
            node (Node): current node
            motion (Node): the motion that current node executes

        Returns:
            flag (bool): True if current node has forced neighbor else False
        """
        x, y, z = node.current
        dx, dy, dz = motion.current
        norm1 = abs(dx) + abs(dy) + abs(dz)
        id = (dx + 1) + 3 * (dy + 1) + 9 * (dz + 1)
        
        if norm1 == 1:
            # 1-d move, check 8 neighbors
            for fn in range(8):
                nx = x + self.jn3d_f1[id][0][fn]
                ny = y + self.jn3d_f1[id][1][fn]
                nz = z + self.jn3d_f1[id][2][fn]
                if (nx, ny, nz) in self.obstacles:
                    return True
        elif norm1 == 2:
            # 2-d move, check 8 neighbors
            for fn in range(8):
                nx = x + self.jn3d_f1[id][0][fn]
                ny = y + self.jn3d_f1[id][1][fn]
                nz = z + self.jn3d_f1[id][2][fn]
                if (nx, ny, nz) in self.obstacles:
                    return True
        elif norm1 == 3:
            # 3-d move, check 6 neighbors
            for fn in range(6):
                nx = x + self.jn3d_f1[id][0][fn]
                ny = y + self.jn3d_f1[id][1][fn]
                nz = z + self.jn3d_f1[id][2][fn]
                if (nx, ny, nz) in self.obstacles:
                    return True
        return False
