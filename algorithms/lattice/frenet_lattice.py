"""
Author: HTY
Email: 1044213317@qq.com
Date: 2023-02-16 16:04
Description:
    - [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
    (https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)
    (https://www.youtube.com/watch?v=Cj6tAQe7UCY)
"""


from quintic_polynomial import QuinticPolynomial
from cubic_spline import CubicSpline2D

import numpy as np
import matplotlib.pyplot as plt
import copy

# Parameter
MAX_V = 20.0 / 3.6  # maximum speed [m/s]
MIN_V = 5 / 3.6     # minimum speed [m/s]
DELTA_V = 5.0 / 3.6  # target speed sampling length [m/s]

MAX_D = 3.0  # maximum road width [m]
DELTA_D = 1.0  # road width sampling length [m]

MAX_T = 8.0  # max prediction time [s]
MIN_T = 2.0  # min prediction time [s]
DELTA_T = 2  # time tick [s]

MAX_S = 30.0
MIN_S = 10.0
DELTA_S = 5.0

MAX_A = 2.0  # maximum acceleration [m/ss]
MAX_K = 1.0  # maximum curvature [1/m]
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class FrenetPath:
    def __init__(self):
        self.index = 0
        self.t = []

        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []    # 用来算曲率
        self.k = []


def calc_frenet_paths(s, s_d, s_dd, d, d_d, d_dd):
    frenet_paths = []
    
    for ti in np.arange(MIN_T, MAX_T + DELTA_T, DELTA_T):       # 轨迹时间
        for di in np.arange(-MAX_D, MAX_D + DELTA_D, DELTA_D):  # 道路宽度
            lat_traj = QuinticPolynomial(d, d_d, d_dd, di, 0.0, 0.0, ti)
            fp = FrenetPath()
            fp.t = [t for t in np.arange(0.0, ti, 0.2)]
            fp.d = [lat_traj.calc_point(t) for t in fp.t]
            fp.d_d = [lat_traj.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_traj.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_traj.calc_third_derivative(t) for t in fp.t]

            for vi in np.arange(MIN_V, MAX_V + DELTA_V, DELTA_V):  # 末端速度
                for si in np.arange(MIN_S, MAX_S + DELTA_S, DELTA_S):      # 道路长度
                    lon_traj = QuinticPolynomial(s, s_d, s_dd, s + si, vi, 0.0, ti)

                    fp_copy = copy.deepcopy(fp)
                    fp_copy.s = [lon_traj.calc_point(t) for t in fp.t]
                    fp_copy.s_d = [lon_traj.calc_first_derivative(t) for t in fp.t]
                    fp_copy.s_dd = [lon_traj.calc_second_derivative(t) for t in fp.t]
                    fp_copy.s_ddd = [lon_traj.calc_third_derivative(t) for t in fp.t]
                    frenet_paths.append(fp_copy)
                    # if check_paths(fp_copy):
                    #     frenet_paths.append(fp_copy)
    return frenet_paths


def calc_cartesian_paths(frenet_path, reference_line):
    # calc global positions
    for i in range(len(frenet_path.s)):
        xi, yi = reference_line.calc_position(frenet_path.s[i])
        if xi is None:
            break
        yaw = reference_line.calc_yaw(frenet_path.s[i])
        di = frenet_path.d[i]
        # frenet to cartesian
        fx = xi + di * np.cos(yaw + np.pi / 2.0)
        fy = yi + di * np.sin(yaw + np.pi / 2.0)
        frenet_path.x.append(fx)
        frenet_path.y.append(fy)

    # calc yaw and ds
    for i in range(len(frenet_path.x) - 1):
        dx = frenet_path.x[i + 1] - frenet_path.x[i]
        dy = frenet_path.y[i + 1] - frenet_path.y[i]
        frenet_path.yaw.append(np.arctan2(dy, dx))
        frenet_path.ds.append(np.hypot(dx, dy))

    frenet_path.yaw.append(frenet_path.yaw[-1])
    frenet_path.ds.append(frenet_path.ds[-1])

    # calc curvature
    for i in range(len(frenet_path.yaw) - 1):
        frenet_path.k.append((frenet_path.yaw[i + 1] - frenet_path.yaw[i]) / frenet_path.ds[i])

    return frenet_path


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((xi - ob[i, 0]) ** 2 + (yi - ob[i, 1]) ** 2)
             for (xi, yi) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])
        if collision:
            return False

    return True


def check_paths(path) -> bool:
    if any([v > MAX_V for v in path.s_d]):  # Max speed check
        return False
    elif any([abs(a) > MAX_A for a in path.s_dd]):  # Max accel check
        return False
    elif any([abs(k) > MAX_K for k in path.k]):  # Max curvature check
        return False
    return True


def frenet_optimal_planning(reference_line: CubicSpline2D, s, s_d, s_dd, d, d_d, d_dd, ob):
    fplist = calc_frenet_paths(s, s_d, s_dd, d, d_d, d_dd)
    path = [calc_cartesian_paths(fp, reference_line) for fp in fplist]

    return path

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x: list, y: list) -> (list, list, list, list, CubicSpline2D):
    reference_line = CubicSpline2D(x, y)
    s_list = np.arange(0, reference_line.s[-1], 0.1)

    x_list, y_list, yaw_list, curvature_list = [], [], [], []
    for s in s_list:
        x, y = reference_line.calc_position(s)
        x_list.append(x)
        y_list.append(y)
        yaw_list.append(reference_line.calc_yaw(s))
        curvature_list.append(reference_line.calc_curvature(s))

    return x_list, y_list, yaw_list, curvature_list, reference_line


def main():
    print(__file__ + " start!!")

    # way points & obstacles
    # way_points_x = [0.0, 10.0, 20.5, 35.0, 70.5]
    # way_points_y = [0.0, -6.0, 5.0, 6.5, 0.0]
    # ob = np.array([[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]])
    way_points_x = [0.0, 40.0]
    way_points_y = [0.0, 40.0]
    ob = np.array([[3.0, 3.0], [10.0, 11.0], [23.0, 25.0]])

    tx, ty, tyaw, tc, reference_line = generate_target_course(way_points_x, way_points_y)

    # initial state
    s = 0.0  # current course position
    s_d = 10.0 / 3.6  # current speed [m/s]
    s_dd = 0.0  # current acceleration [m/ss]
    d = 0.0  # current lateral position [m]
    d_d = 0.0  # current lateral speed [m/s]
    d_dd = 0.0  # current lateral acceleration [m/s]

    area = 30.0  # animation area length [m]

    while True:
        # path = frenet_optimal_planning(reference_line, s, s_d, s_dd, d, d_d, d_dd, ob)
        #
        # s = path.s[1]
        # d = path.d[1]
        # d_d = path.d_d[1]
        # d_dd = path.d_dd[1]
        # s_d = path.s_d[1]
        # s_dd = path.s_dd[1]
        #
        # if np.hypot(path.z[1] - tx[-1], path.y[1] - ty[-1]) <= 2.0:
        #     print("Goal")
        #     break

        candidate_trajectories = frenet_optimal_planning(reference_line, s, s_d, s_dd, d, d_d, d_dd, ob)
        print(len(candidate_trajectories))
        # if show_animation:  # pragma: no cover
        for path in candidate_trajectories:
            # plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-r")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(s_d * 3.6)[0:4])
            plt.grid(True)
            # plt.pause(0.0001)
        plt.show()

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
