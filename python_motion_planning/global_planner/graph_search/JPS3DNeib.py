

class JPS3DNeib:
    def __init__(self):
        self.ns = [[[[] for _ in range(26)] for _ in range(3)] for _ in range(27)]
        self.f1 = [[[[] for _ in range(12)] for _ in range(3)] for _ in range(27)]
        self.f2 = [[[[] for _ in range(12)] for _ in range(3)] for _ in range(27)]
        self.nsz = [[26, 0], [1, 8], [3, 12], [7, 12]]  
        id = 0
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    norm1 = abs(dx) + abs(dy) + abs(dz)
                    for dev in range(self.nsz[norm1][0]):
                        self.ns[id][0][dev], self.ns[id][1][dev], self.ns[id][2][dev] = self.Neib(dx, dy, dz, norm1, dev)
                    for dev in range(self.nsz[norm1][1]):
                        self.f1[id][0][dev], self.f1[id][1][dev], self.f1[id][2][dev], self.f2[id][0][dev], self.f2[id][1][dev], self.f2[id][2][dev] = self.FNeib(dx, dy, dz, norm1, dev)
                    id += 1

    def Neib(self, dx, dy, dz, norm1, dev):
        if norm1 == 0:
            neighbors = [
                (1, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 1, 0), (-1, 1, 0), 
                (0, -1, 0), (1, -1, 0), (-1, -1, 0), (0, 0, 1), (1, 0, 1),
                (-1, 0, 1), (0, 1, 1), (1, 1, 1), (-1, 1, 1), (0, -1, 1),
                (1, -1, 1), (-1, -1, 1), (0, 0, -1), (1, 0, -1), (-1, 0, -1),
                (0, 1, -1), (1, 1, -1), (-1, 1, -1), (0, -1, -1), (1, -1, -1),
                (-1, -1, -1)
            ]
            return neighbors[dev]
        elif norm1 == 1:
            return dx, dy, dz
        elif norm1 == 2:
            if dev == 0:
                if dz == 0:
                    return 0, dy, 0
                else:
                    return 0, 0, dz
            elif dev == 1:
                if dx == 0:
                    return 0, dy, 0
                else:
                    return dx, 0, 0
            elif dev == 2:
                return dx, dy, dz
        elif norm1 == 3:
            neighbors = [
                (dx, 0, 0), (0, dy, 0), (0, 0, dz), (dx, dy, 0), (dx, 0, dz),
                (0, dy, dz), (dx, dy, dz)
            ]
            return neighbors[dev]

    def FNeib(self, dx, dy, dz, norm1, dev):
        if norm1 == 1:
            neighbors = [
                (0, 1, 0), (0, -1, 0), (1, 0, 0), (1, 1, 0), (1, -1, 0),
                (-1, 0, 0), (-1, 1, 0), (-1, -1, 0)
            ]
            fx, fy, fz = neighbors[dev]
            nx, ny, nz = fx, fy, dz
            if dx != 0:
                fz, fx = fx, 0
                nz, nx = fz, dx
            if dy != 0:
                fz, fy = fy, 0
                nz, ny = fz, dy
            return fx, fy, fz, nx, ny, nz
        elif norm1 == 2:
            if dx == 0:
                f_neighbors = [
                    (0, 0, -dz), (0, -dy, 0), (1, 0, 0), (-1, 0, 0),
                    (1, 0, -dz), (1, -dy, 0), (-1, 0, -dz), (-1, -dy, 0),
                    (1, 0, 0), (1, 0, 0), (-1, 0, 0), (-1, 0, 0)
                ]
                n_neighbors = [
                    (0, dy, -dz), (0, -dy, dz), (1, dy, dz), (-1, dy, dz),
                    (1, dy, -dz), (1, -dy, dz), (-1, dy, -dz), (-1, -dy, dz),
                    (1, dy, 0), (1, 0, dz), (-1, dy, 0), (-1, 0, dz)
                ]
                fx, fy, fz = f_neighbors[dev]
                nx, ny, nz = n_neighbors[dev]
                return fx, fy, fz, nx, ny, nz
            elif dy == 0:
                f_neighbors = [
                    (0, 0, -dz), (-dx, 0, 0), (0, 1, 0), (0, -1, 0), 
                    (0, 1, -dz), (-dx, 1, 0), (0, -1, -dz), (-dx, -1, 0),
                    (0, 1, 0), (0, 1, 0), (0, -1, 0), (0, -1, 0)
                ]
                n_neighbors = [
                    (dx, 0, -dz), (-dx, 0, dz), (dx, 1, dz), (dx, -1, dz),
                    (dx, 1, -dz), (-dx, 1, dz), (dx, -1, -dz), (-dx, -1, dz),
                    (dx, 1, 0), (0, 1, dz), (dx, -1, 0), (0, -1, dz)
                ]
                fx, fy, fz = f_neighbors[dev]
                nx, ny, nz = n_neighbors[dev]
                return fx, fy, fz, nx, ny, nz
            else:
                f_neighbors = [
                    (0, -dy, 0), (-dx, 0, 0), (0, 0, 1), (0, 0, -1),
                    (0, -dy, 1), (-dx, 0, 1), (0, -dy, -1), (-dx, 0, -1),
                    (0, 0, 1), (0, 0, 1), (0, 0, -1), (0, 0, -1)
                ]
                n_neighbors = [
                    (dx, -dy, 0), (-dx, dy, 0), (dx, dy, 1), (dx, dy, -1),
                    (dx, -dy, 1), (-dx, dy, 1), (dx, -dy, -1), (-dx, dy, -1),
                    (dx, 0, 1), (0, dy, 1), (dx, 0, -1), (0, dy, -1)]
                fx, fy, fz = f_neighbors[dev]
                nx, ny, nz = n_neighbors[dev]
                return fx, fy, fz, nx, ny, nz
        elif norm1 == 3:
            f_neighbors = [
                (-dx, 0, 0), (0, -dy, 0), (0, 0, -dz), (0, -dy, -dz),
                (-dx, 0, -dz), (-dx, -dy, 0), (-dx, 0, 0), (-dx, 0, 0),
                (0, -dy, 0), (0, -dy, 0), (0, 0, -dz), (0, 0, -dz)
            ]
            n_neighbors = [
                (-dx, dy, dz), (dx, -dy, dz), (dx, dy, -dz), (dx, -dy, -dz), 
                (-dx, dy, -dz), (-dx, -dy, dz), (-dx, 0, dz), (-dx, dy, 0), 
                (0, -dy, dz), (dx, -dy, 0), (0, dy, -dz), (dx, 0, -dz)
            ]
            fx, fy, fz = f_neighbors[dev]
            nx, ny, nz = n_neighbors[dev]
            return fx, fy, fz, nx, ny, nz