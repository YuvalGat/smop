from tqdm import tqdm, trange

from utils import *
import pandas as pd

MASS = 0.005
SHRAPNEL_DENSITY = 19300
THICKNESS = 0.01
CUT_AREA = np.pi * np.power(3 * MASS / (4 * np.pi * SHRAPNEL_DENSITY), 2 / 3)
C = 6.475
ALPHA = 0.889
BETA = -0.945
GAMMA = 1.262
LAMBDA = 0.019
DENSITY_AIR = 1.225
BODY_CONSTANT = 0.752
CD = 0.9

"""
a ---- c
|      |
b ---- d
"""

warhead_data = pd.read_csv('data.csv')
ANGLES = np.array(warhead_data['angle'])
DENSITIES = np.array(warhead_data['density'])
VELOCITIES = np.array(warhead_data['velocity'])


def f(E):
    return 2 / np.pi * np.arctan(0.00481 * E)


class Explosion:
    def __init__(self, origin, fissure, direction, sdf, svf):
        self.origin = origin.astype(float)
        self.fissure = fissure
        self.direction = normalize(direction.astype(float))
        self.sdf = sdf  # Shrapnel density function
        self.svf = svf  # Shrapnel velocity function

    def explode_on(self, vertices, missile, lam=None):
        theta = get_angle_off_vertex(self.origin, self.direction, vertices[0])
        return explode_on(self.origin, np.array(vertices), self.fissure, missile.warhead_center,
                          missile.warhead_radius, missile.direction, self.sdf(theta), self.svf(theta), lam)

    def explode_on_split_surface(self, split_vertices, missile, lams=None):
        if lams is None:
            all_data = np.array(
                [[np.array(self.explode_on(v, missile)) for v in row] for row in split_vertices])
        else:
            all_data = np.array(
                [[np.array(self.explode_on(v, missile, lams[j][i])) for i, v in enumerate(row)] for j, row in
                 enumerate(split_vertices)])
        hit_dist, vel, cosangles, distances, lams = all_data[:, :, 0], all_data[:, :, 1], all_data[:, :, 2], \
                                                    all_data[:, :, 3], all_data[:, :, 4]
        return hit_dist, vel, cosangles, distances, lams

    def plot_explosion(self, ax):
        explosion_missile = Missile(self.origin - self.direction, self.direction, 0.7, 0.2, 0.3)
        explosion_missile.plot_missile(ax, c='b')


class Missile:
    def __init__(self, warhead_center, direction, warhead_length, warhead_radius, homing_head_length):
        self.warhead_center = warhead_center.astype(float)
        self.direction = normalize(direction.astype(float))
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


@jit(nopython=True)
def sdf(theta):
    return 4 * np.pi * DENSITIES[np.argmax(ANGLES >= (180 / np.pi * theta))]


@jit(nopython=True)
def svf(theta):
    return VELOCITIES[np.argmax(ANGLES >= (180 / np.pi * theta))]


@jit(nopython=True)
def get_total_energy_penetrated(hit_dist, vel, cosangles, distances):
    total_energy = 0
    for i, row in enumerate(hit_dist):
        for j, hit_count in enumerate(row):
            vel_shot = vel[i][j]
            distance_flown = distances[i][j]
            interception_angle = cosangles[i][j]
            vel_hit = get_velocity_after_flight(vel_shot, CD, DENSITY_AIR, SHRAPNEL_DENSITY, BODY_CONSTANT, MASS,
                                                distance_flown)
            vel_after_penetration = get_penetration_velocity(vel_hit, CUT_AREA, THICKNESS, MASS, interception_angle,
                                                             C, ALPHA, BETA, GAMMA, LAMBDA)
            total_energy += 0.5 * hit_count * MASS * np.power(vel_after_penetration, 2)

    return total_energy


def get_total_energy_deterministic(vel, cosangles, distances, lams):
    vels_hit = get_velocity_after_flight(vel, CD, DENSITY_AIR, SHRAPNEL_DENSITY, BODY_CONSTANT, MASS, distances)
    vels_after_penetration = get_penetration_velocity(vels_hit, CUT_AREA, THICKNESS, MASS, cosangles, C, ALPHA, BETA,
                                                      GAMMA, LAMBDA)
    if DISPLAY_GRAPHS:
        plot_graph(vels_hit, 'VELS BEFORE')
        plot_graph(vels_after_penetration, 'VELS AFTER')
    return 0.5 * MASS * np.sum(lams * np.power(vels_after_penetration, 2))


def get_energy_matrix(vel, distances, cosangles):
    vels_hit = get_velocity_after_flight(vel, CD, DENSITY_AIR, SHRAPNEL_DENSITY, BODY_CONSTANT, MASS, distances)
    vels_after_penetration = get_penetration_velocity(vels_hit, CUT_AREA, THICKNESS, MASS, cosangles, C, ALPHA, BETA,
                                                      GAMMA, LAMBDA)
    return 0.5 * MASS * np.power(vels_after_penetration, 2)


def get_interception_probability(explosion, missile, accuracy):
    vertices = missile.get_projection_coordinates(explosion)['warhead_coords']
    split_vertices = split_rectangle(vertices, accuracy, accuracy)
    _, vel, cosangles, distances, lams = explosion.explode_on_split_surface(split_vertices, m)
    em = get_energy_matrix(vel, distances, cosangles)
    return get_total_probability(lams, em, f)


if __name__ == '__main__':
    # This simulates a 5gr ball of tungsten, penetrating tough steel
    SHOW_3D = True
    DISPLAY_GRAPHS = False
    fig = plt.figure()
    acc = 50
    explosion = Explosion(np.array([2, -5, 0]), 4 * np.pi, np.array([0, 1, -1]),
                          sdf, svf)
    m = Missile(np.array([0, 0, 1]), np.array([-1, -1, 0]), 2, 0.3, 1)
    vertices = m.get_projection_coordinates(explosion)['warhead_coords']
    split_vertices = split_rectangle(vertices, acc, acc)
    Es = []
    hit_dist, vel, cosangles, distances, lams = explosion.explode_on_split_surface(split_vertices, m)
    # for i in trange(100):
    #     hit_dist, vel, cosangles, distances, _ = explosion.explode_on_split_surface(split_vertices, m, lams)
    #     E = get_total_energy_penetrated(hit_dist, vel, cosangles, distances)
    #     Es.append(E)
    # plt.hist(Es)
    # plt.show()
    # print(np.mean(Es))
    # print(Es)
    # print(get_total_energy_deterministic(vel, cosangles, distances, lams))
    if DISPLAY_GRAPHS:
        if SHOW_3D:
            ax = plt.axes(projection='3d')
            m.plot_missile(ax)
            explosion.plot_explosion(ax)
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            ax.set_zlim(-10, 10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.show()
        plot_graph(hit_dist, 'Hit distribution')
        plot_graph(vel, 'Vels upon leaving warhead')
        plot_graph(cosangles, 'Cos of hit angles')
        plot_graph(distances, 'Distances')
        plot_graph(lams, 'Lambdas')
        vels_hit = get_velocity_after_flight(vel, CD, DENSITY_AIR, SHRAPNEL_DENSITY, BODY_CONSTANT, MASS, distances)
        vels_after_penetration = get_penetration_velocity(vels_hit, CUT_AREA, THICKNESS, MASS, cosangles, C, ALPHA,
                                                          BETA, GAMMA, LAMBDA)
        plot_graph(vels_hit, 'Vels when hitting missile')
        plot_graph(vels_after_penetration, 'Vels after penetration')
    X_ACC = np.linspace(10, 1000, 50)
    Y = [get_interception_probability(explosion, m, int(a)) for a in X_ACC]
    plt.plot(X_ACC, Y)
    plt.show()
    # e = np.array([0, 5, 3])
    # m = np.array([1, -1, 0])
    # d = np.array([0, 1, 0])
    # r = 2
    # h = get_hit_location(e, m, d, r)
    # n = h + normalize(perpendicular_component(get_hit_location(e, m, d, r), d))
    # M = Missile(np.array([0, 0, 0]), d, 5, 2, 2)
    # ax.scatter(e[0], e[1], e[2], c='k')
    # ax.scatter(m[0], m[1], m[2], c='b')
    # ax.scatter(h[0], h[1], h[2], c='m', s=50)
    # ax.scatter(n[0], n[1], n[2], c='r')
    # print(np.arccos(get_hit_angle_cos(e, m, d, r)) * 180 / np.pi)
    # # M.plot_missile(ax)
    # ax.set_xlabel('x')
    # plt.show()
