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


if __name__ == '__main__':
    origin = np.array([0, 0, 0])
    vertices = [np.array([-1, 1, 1]), np.array([-1, -1, 1]), np.array([1, 1, 1]), np.array([1, -1, 1])]
    a, b, c, d = vertices
    N = 6e9

    acc = 100
    split_vertices = split_rectangle(vertices, acc, acc)
    lams = [[get_poisson_parameter_on_rectangle(origin, v, 4 * np.pi, N) for v in split_vertices[row]] for row in
            range(acc)]
    print(get_solid_angle_by_triangular_surface(vertices[0], vertices[1], vertices[2]))
    # dist = np.add(0, np.array([poisson(lams) for i in range(1000)]).sum(axis=0))
    dist = poisson(lams)
    plt.imshow(dist)
    plt.colorbar()
    plt.show()
    print(np.sum(dist))
