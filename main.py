import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from numba import njit

"""
a ---- c
|      |
b ---- d
"""

poisson = np.random.poisson


# @njit
def get_solid_angle_by_triangular_surface(a, b, c):
    triple_product = np.linalg.det(np.dstack([a, b, c]))[0]
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
        dist = np.rot90(dist)
        return dist


class Missile:
    def __init__(self, warhead_center, direction, warhead_length, warhead_radius, homing_head_length):
        self.warhead_center = warhead_center
        self.direction = direction / np.linalg.norm(direction)
        self.warhead_length = warhead_length
        self.warhead_radius = warhead_radius
        self.homing_head_length = homing_head_length

    def get_projection_coordinates(self, explosion):
        c = self.warhead_center
        v = explosion.origin - c
        u = self.direction
        perp = v - np.dot(v, u) * u
        perp /= np.linalg.norm(perp)
        p = np.cross(perp, u) * self.warhead_radius
        l = u * self.warhead_length / 2
        return {'warhead_coords': [c + p + l, c - p + l, c + p - l, c - p - l],
                'homing_head_coords': [c + p + l, c - p + l, c + l + u * self.homing_head_length]}


def get_minimal_penetration_velocity(m, A, d, theta, b, n):
    # Ricckihazzi formula
    return b / np.sqrt(m) * np.sqrt(np.power(A, 1.5) * np.power(d / (np.sqrt(A) * np.cos(theta)), n))


def get_penetration_velocity(vs, A, d, m, theta, C, alpha, beta, gamma, lam):
    # Thor formula
    return vs - np.pow(10, C) * np.power(A * d, alpha) * np.power(m, beta) * np.power(vs, lam) / np.power(np.cos(theta),
                                                                                                          gamma)


if __name__ == '__main__':
    origin = np.array([10, -1, 1])
    N = 6e9

    acc = 50
    explosion = Explosion(origin, 4 * np.pi, N)
    m = Missile(np.array([0, 0, 1]), np.array([-1, 0, 0]), 2, 0.3, 1)
    vertices = m.get_projection_coordinates(explosion)['warhead_coords']
    print(vertices)
    split_vertices = split_rectangle(vertices, acc, acc)
    dist = explosion.explode_on_split_surface(split_vertices)
    plt.imshow(dist)
    plt.colorbar()
    plt.show()
    print(np.sum(dist))
