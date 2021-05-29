from utils import *

"""
a ---- c
|      |
b ---- d
"""

poisson = np.random.poisson


class Explosion:
    def __init__(self, origin, fissure, shrapnel_count, direction, sdf, svf):
        self.origin = origin
        self.fissure = fissure
        self.shrapnel_count = shrapnel_count
        self.direction = normalize(direction)
        self.sdf = sdf  # Shrapnel density function
        self.svf = svf  # Shrapnel velocity function

    def explode_on(self, vertices):
        theta = self.get_angle_off_vertex(vertices[0])
        lam = get_poisson_parameter_on_rectangle(self.origin, vertices, self.fissure, self.shrapnel_count) * self.sdf(
            theta)
        return poisson(lam), self.svf(theta)

    def get_angle_off_vertex(self, vertex):
        theta = np.arcsin(np.dot(vertex - self.origin, self.direction) / np.linalg.norm(vertex - self.origin))
        if theta < 0:
            theta += np.pi
        return theta

    def explode_on_split_surface(self, split_vertices):
        dist = [[self.explode_on(v)[0] for v in row] for row in split_vertices]
        dist = np.rot90(dist)
        return dist

    def plot_explosion(self, ax):
        explosion_missile = Missile(self.origin - self.direction, self.direction, 0.7, 0.2, 0.3)
        explosion_missile.plot_missile(ax, c='b')


class Missile:
    def __init__(self, warhead_center, direction, warhead_length, warhead_radius, homing_head_length):
        self.warhead_center = warhead_center
        self.direction = normalize(direction)
        self.warhead_length = warhead_length
        self.warhead_radius = warhead_radius
        self.homing_head_length = homing_head_length

    def get_projection_coordinates(self, explosion):
        c = self.warhead_center
        v = explosion.origin - c
        u = self.direction
        p = np.cross(normalize(perpendicular_component(v, u)), u) * self.warhead_radius
        l = u * self.warhead_length / 2
        return {'warhead_coords': [c + p + l, c - p + l, c + p - l, c - p - l],
                'homing_head_coords': [c + p + l, c - p + l, c + l + u * self.homing_head_length]}

    def get_missile_coordinates(self):
        u = self.direction
        c = self.warhead_center
        homing_head = c + u * (self.warhead_length + self.homing_head_length)
        x = [homing_head[0]]
        y = [homing_head[1]]
        z = [homing_head[2]]
        first_vector = normalize(perpendicular_vector(u))
        second_vector = normalize(np.cross(first_vector, u))
        n = 20
        for theta in np.linspace(0, np.pi * 2, n):
            point = c + u * self.warhead_length + (
                    first_vector * np.cos(theta) + second_vector * np.sin(theta)) * self.warhead_radius
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        for theta in np.linspace(0, np.pi * 2, n):
            point = c - u * self.warhead_length + (
                    first_vector * np.cos(theta) + second_vector * np.sin(theta)) * self.warhead_radius
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        triangles = np.asarray([[0, i, i + 1] for i in range(1, n)] + [[i + 1, i, i + n] for i in range(1, n)] +
                               [[i + 1, i + n, i + n + 1] for i in range(1, n)] + [[i, i + 1, i + int(n / 2)] for i in
                                                                                   range(n, int(3 * n / 2) + 1)] +
                               [[i, i + int(n / 2), i + int(n / 2) + 1] for i in range(n, int(3 * n / 2))])
        tri = mtri.Triangulation(x, y, triangles)
        return tri, z

    def plot_missile(self, ax, c='r'):
        tri, z = self.get_missile_coordinates()
        ax.plot_trisurf(tri, z, color=c)


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


def get_penetration_velocity(vs, A, d, m, theta, C, alpha, beta, gamma, lam):
    """
    Thor formula
    :param vs: Entry velocity in m/s
    :param A: Cross-area of fragment in cm^2
    :param d: Penetrated surface thickness in mm
    :param m: Mass of fragment in gram
    :param theta: Penetration angle
    :return: Velocity of exiting fragment
    """
    m_grain = m / 15.432
    A_in2 = A / 6452
    d_in = d / 25.4
    return vs - np.power(10, C) * np.power(A_in2 * d_in, alpha) * np.power(m_grain, beta) * np.power(vs,
                                                                                                     lam) / np.power(
        np.cos(theta), gamma)


def sdf(theta):
    if np.pi / 6 < theta < np.pi / 4:
        return 1
    else:
        return 0


if __name__ == '__main__':
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    origin = np.array([0, -0.3, 1])
    N = 1000e4

    acc = 50
    explosion = Explosion(origin, 4 * np.pi, N, np.array([0.3, 1, 0]),
                          sdf, lambda x: 10)
    m = Missile(np.array([0, 0, 1]), np.array([-1, 0, 0]), 2, 0.3, 1)
    m.plot_missile(ax)
    explosion.plot_explosion(ax)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    plt.show()
    vertices = m.get_projection_coordinates(explosion)['warhead_coords']
    split_vertices = split_rectangle(vertices, acc, acc)
    dist = explosion.explode_on_split_surface(split_vertices)
    plt.imshow(dist)
    plt.colorbar()
    plt.show()
