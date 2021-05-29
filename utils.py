import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from numba import njit


def normalize(v):
    if not any(v):
        return v
    else:
        return v / np.linalg.norm(v)


def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


def perpendicular_component(v, d):
    """
    returns the perpendicular component of v in respect to d
    """
    d = normalize(d)
    return v - np.dot(v, d) * d


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


def get_hit_location(e, m, d, r):
    """
    get the point where the line between e and m intersects with the cylinder with radius r and
    its center is (0, 0, 0) pointing in direction d
    :param e: the location of the explosion
    :param m: the location where the shrapnel hit the missile rectangle
    :param d: the direction of the missile
    :param r: the radius of the missile
    :return:
    """
    e = perpendicular_component(e, d)
    m = perpendicular_component(m, d)
    a = np.linalg.norm(m)**2
    b = np.dot(e, m)**2
    c = np.linalg.norm(e - m)**2
    x = (a - b + ((a - b)**2 + (r**2 - a)*c)**0.5) / c
    return x * e + (1 - x) * m


def get_hit_angle_cos(e, m, d, r):
    """
    :param e: the location of the explosion
    :param m: the location where the shrapnel hit the missile rectangle
    :param d: the direction of the missile
    :param r: the radius of the missile
    :return:
    """
    return np.dot(normalize(get_hit_location(e, m, d, r)), normalize(e - m))


if __name__ == "__main__":
    e = np.array([2, 0, 5])
    m = np.array([0, 2, -1])
    d = np.array([0, 0, 1])
    r = 2
    h = get_hit_location(e, m, d, r)
    print(h)
    plt.plot(e[1], e[0], 'k.')
    plt.plot(m[1], m[0], 'k.')
    plt.plot(h[1], h[0], 'r.')
    plt.plot(r * np.cos(np.arange(0, 2*np.pi, 0.01)), r * np.sin(np.arange(0, 2*np.pi, 0.01)), 'm')
    plt.show()
