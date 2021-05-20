import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

"""
a ---- c
|      |
b ---- d
"""

poisson = np.random.poisson


def get_solid_angle_by_triangular_surface(a, b, c):
    triple_product = np.dot(a, np.cross(b, c))
    a_size = np.linalg.norm(a)
    b_size = np.linalg.norm(b)
    c_size = np.linalg.norm(c)
    tan_half = triple_product / (
            a_size * b_size * c_size + np.dot(a, b) * c_size + np.dot(a, c) * b_size + np.dot(b, c) * a_size)
    omega = 2 * np.arctan(tan_half)

    return np.abs(omega)


def get_poisson_parameter_on_rectangle(origin, vertices, explosion_fissure, shrapnel_count):
    """

    :param origin: Origin of the explosion.
    :type origin: np.ndarray
    :param vertices: Rectangle vertices.
    :type vertices: [np.ndarray]
    :param explosion_fissure: Solid angle corresponding to the explosion.
    :type explosion_fissure: float
    :param shrapnel_count: The number of shrapnels in the explosion.
    :type shrapnel_count: int
    :return: Corresponding lambda.
    """
    a, b, c, d = [v - origin for v in vertices]
    solid_angle = get_solid_angle_by_triangular_surface(a, b, c) + get_solid_angle_by_triangular_surface(b, c, d)
    lam = shrapnel_count * solid_angle / explosion_fissure

    return lam


def split_rectangle(vertices, w, h):
    a, b, c, d = vertices
    x_split = np.linspace(a, c, w + 1)
    y_split = np.linspace(a, b, h + 1)
    split_vertices = []
    for i in range(w):
        row = []
        for j in range(h):
            sub_vertices = [x_split[i] + y_split[j] - a, x_split[i] + y_split[j + 1] - a,
                            x_split[i + 1] + y_split[j] - a, x_split[i + 1] + y_split[j + 1] - a]
            row.append(sub_vertices)
        split_vertices.append(row)

    return split_vertices


class Explosion:
    def __init__(self, origin, fissure, shrapnel_count):
        self.origin = origin
        self.fissure = fissure
        self.shrapnel_count = shrapnel_count

    def explode_on(self, vertices):
        lam = get_poisson_parameter_on_rectangle(self.origin, vertices, self.fissure, self.shrapnel_count)
        return poisson(lam)

    def explode_on_split_surface(self, split_vertices):
        dist = [[self.explode_on(v) for v in row] for row in split_vertices]
        return dist


if __name__ == '__main__':
    origin1 = np.array([1, 1, 0])
    origin2 = np.array([0, 0, 0])
    vertices = [np.array([-1, 1, 1]), np.array([-1, -1, 1]), np.array([1, 1, 1]), np.array([1, -1, 1])]
    a, b, c, d = vertices
    N = 6e9

    acc = 100
    explosion1 = Explosion(origin1, 4 * np.pi, N)
    explosion2 = Explosion(origin2, 2 * np.pi, N / 2)
    split_vertices = split_rectangle(vertices, acc, acc)

    dist1 = explosion1.explode_on_split_surface(split_vertices)
    dist2 = explosion2.explode_on_split_surface(split_vertices)
    dist = np.add(dist1, dist2)
    print(get_solid_angle_by_triangular_surface(vertices[0], vertices[1], vertices[2]))
    plt.imshow(dist)
    plt.colorbar()
    plt.show()
    print(np.sum(dist))
