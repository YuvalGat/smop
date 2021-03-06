import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from numba import jit

poisson = np.random.poisson


@jit(nopython=True)
def normalize(v: np.array([float])):
    return v / np.linalg.norm(v)


def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])


@jit(nopython=True)
def perpendicular_component(v, d):
    """
    returns the perpendicular component of v in respect to d, d must be normalized
    """
    return v - np.dot(v, d) * d


@jit(nopython=True)
def get_solid_angle_by_triangular_surface(a, b, c):
    triple_product = np.dot(a, np.cross(b, c))
    a_size = np.linalg.norm(a)
    b_size = np.linalg.norm(b)
    c_size = np.linalg.norm(c)
    tan_half = triple_product / (
            a_size * b_size * c_size + np.dot(a, b) * c_size + np.dot(a, c) * b_size + np.dot(b, c) * a_size)
    omega = 2 * np.arctan(tan_half)

    return np.abs(omega)


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


@jit(nopython=True)
def get_hit_location(e, m, d, r):
    """
    get the point where the line between e and m intersects with the cylinder with radius r and
    its center is (0, 0, 0) pointing in direction d
    :param e: the location of the explosion
    :param m: the location where the shrapnel hit the missile rectangle
    :param d: the direction of the missile, must be normalized
    :param r: the radius of the missile
    :return:
    """
    p_e = perpendicular_component(e, d)
    p_m = perpendicular_component(m, d)
    a = np.dot(p_m, p_m)
    b = np.dot(p_e, p_m) ** 2
    c = np.dot(p_e - p_m, p_e - p_m)
    x = (a - b + ((a - b) ** 2 + (r ** 2 - a) * c) ** 0.5) / c
    return x * e + (1 - x) * m


@jit(nopython=True)
def get_hit_angle_cos(e, m, d, r):
    """
    :param e: the location of the explosion
    :param m: the location where the shrapnel hit the missile rectangle
    :param d: the direction of the missile, must be normalized
    :param r: the radius of the missile
    :return:
    """
    return np.dot(normalize(perpendicular_component(get_hit_location(e, m, d, r), d)), normalize(e - m))


def plot_graph(matrix, title):
    copy = np.rot90(matrix)
    plt.imshow(copy)
    plt.title(title)
    plt.colorbar()
    plt.show()


def get_minimal_penetration_velocity(m, A, d, theta, b, n):
    """
    Ricckihazzi formula
    :param m: Mass of fragment in gram
    :param A: Cross-area of fragment in mm^2
    :param d: Penetrated surface thickness in mm
    :param theta: Penetration angle
    :return: Minimal velocity required to penetrate surface
    """
    return b / np.sqrt(m) * np.sqrt(np.power(np.power(A, 1.5) * (d / (np.sqrt(A) * np.cos(theta))), n))


@jit(nopython=True)
def get_penetration_velocity(vs, A, d, m, costheta, C, alpha, beta, gamma, lam):
    """
    Thor formula
    :param vs: Entry velocity in m/s
    :param A: Cross-area of fragment in m^2
    :param d: Penetrated surface thickness in m
    :param m: Mass of fragment in kg
    :param costheta: Cosine of penetration angle
    :return: Velocity of exiting fragment in m/s
    """
    vs_fs = vs * 3.281
    m_grain = m * 15432
    A_in2 = A * 1550
    d_in = d * 39.37
    return np.maximum(
        (vs_fs - np.power(10, C) * np.power(A_in2 * d_in, alpha) * np.power(m_grain, beta) * np.power(vs_fs,
                                                                                                      lam) / np.power(
            costheta, gamma)) / 3.281, 0)


@jit(nopython=True)
def get_velocity_after_flight(v0, Cd, rho, rho_f, k_s, m, x):
    return v0 * np.exp(-Cd * rho / (2 * np.power(np.power(rho_f * k_s, 2) * m, 1 / 3)) * x)


@jit(nopython=True)
def get_angle_off_vertex(origin, direction, vertex):
    theta = np.arccos(np.dot(vertex - origin, direction) / np.linalg.norm(vertex - origin))
    if theta < 0:
        theta += np.pi
    return theta


@jit(nopython=True)
def explode_on(origin, vertices, fissure, warhead_center, warhead_radius, missile_direction, sdf_theta, svf_theta,
               lam=None):
    if lam is None:
        a, b, c, d = [v - origin for v in vertices]
        solid_angle = get_solid_angle_by_triangular_surface(a, b, c) + get_solid_angle_by_triangular_surface(b, c, d)
        lam = sdf_theta * solid_angle / fissure
    rel_origin = origin - warhead_center
    rel_hit = vertices[0] - warhead_center
    radius = warhead_radius
    direction = missile_direction
    rel_real_hit = get_hit_location(rel_origin, rel_hit, direction, radius)
    return poisson(lam), svf_theta, get_hit_angle_cos(rel_origin, rel_hit, direction, radius), np.linalg.norm(
        rel_origin - rel_real_hit), lam


def get_total_probability(lam, energy, f):
    return 1 - np.exp(-np.sum(lam * f(energy)))


if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    e = np.array([-5, 0, 3])
    m = np.array([1, 1, 0])
    d = np.array([1, 0, 0])
    r = 2
    h = get_hit_location(e, m, d, r)
    ax.scatter(e[1], e[0], e[2], c='k')
    ax.scatter(m[1], m[0], m[2], c='b')
    ax.scatter(h[1], h[0], h[2], c='m')
    # print(np.arccos(get_hit_angle_cos(e, m, d, r)) * 180 / np.pi)
    # plt.plot(e[1], e[0], 'k.')
    # plt.plot(m[1], m[0], 'k.')
    # plt.plot(h[1], h[0], 'r.')
    # plt.plot(r * np.cos(np.arange(0, 2*np.pi, 0.01)), r * np.sin(np.arange(0, 2*np.pi, 0.01)), 'm')
    plt.show()
