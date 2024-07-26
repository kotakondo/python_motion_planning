"""
@file: node.py
@breif: 2-dimension node data stucture
@author: Yang Haodong, Wu Maojia
@update: 2024.3.15
"""

class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost

    Examples:
        >>> from env import Node
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 5), 2, 8)
        >>> node3 = Node((2, 0), (1, 6), 3, 1)
        ...
        >>> node1 + node2
        >>> Node((2, 0), (2, 3), 3, 2)
        ...
        >>> node1 == node2
        >>> True
        ...
        >>> node1 != node3
        >>> True
    """
    def __init__(self, current: tuple, parent = None, g: float = 0, h: float = 0, weight: float = 1.0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
        self.weight = weight
    
    def __add__(self, node):
        assert isinstance(node, Node)
        assert self.weight == node.weight # weight should be the same
        return Node((self.x + node.x, self.y + node.y), self.parent, self.g + node.g, self.h, self.weight)

    def __eq__(self, node) -> bool:
        if not isinstance(node, Node):
            return False
        assert self.weight == node.weight # weight should be the same
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        assert self.weight == node.weight # weight should be the same
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        assert isinstance(node, Node)
        assert self.weight == node.weight # weight should be the same
        return self.f < node.f or (self.f == node.f and self.f - self.g < node.f - node.g)

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