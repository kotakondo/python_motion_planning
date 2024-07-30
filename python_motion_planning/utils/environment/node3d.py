"""
@file: node.py
@breif: 3-dimension node data stucture
@author: Yang Haodong, Wu Maojia, Kota Kondo
@update: 2024.7.26
"""

class Node3D(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost
        weight (float): weight of heuristic cost
    """
    def __init__(self, current: tuple, parent = None, g: float = 0, h: float = 0, weight: float = 1.0, dx: int = 0, dy: int = 0, dz: int = 0, wait_time: float = 0.0) -> None:

        assert len(current) == 3, "current should be 3-dimension tuple"

        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
        self.weight = weight
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.wait_time = wait_time

    def __add__(self, node):
        assert isinstance(node, Node3D)
        assert self.weight == node.weight # weight should be the same
        return Node3D((self.x + node.x, self.y + node.y, self.z + node.z), self.parent, self.g + node.g, self.h, self.weight, wait_time = self.wait_time + node.wait_time)

    def __eq__(self, node) -> bool:
        if not isinstance(node, Node3D):
            return False
        assert self.weight == node.weight # weight should be the same
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        assert self.weight == node.weight # weight should be the same
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        assert isinstance(node, Node3D)
        assert self.weight == node.weight # weight should be the same
        return self.f < node.f or (self.f == node.f and self.weight * self.h < node.weight * node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {}, {})\n".format(self.current, self.parent, self.g, self.h, self.f)

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def f(self) -> float:
        return self.g + self.weight * self.h
    
    @property
    def x(self) -> float:
        return self.current[0]
    
    @property
    def y(self) -> float:
        return self.current[1]
    
    @property
    def z(self) -> float:
        return self.current[2]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent.current[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent.current[1]
        else:
            return None
        
    @property
    def pz(self) -> float:
        if self.parent:
            return self.parent.current[2]
        else:
            return None
        