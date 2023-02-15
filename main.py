"""
Description: 
version: v1.0
Author: HTY
Date: 2023-01-29 21:53:19
"""

from frenet_optimal_trajectory import *

import numpy as np

render = True

def main():
    print(__file__ + " start!!")

    # way points & obstacles
    # way_points_x = [0.0, 10.0, 20.5, 35.0, 70.5]
    # way_points_y = [0.0, -6.0, 5.0, 6.5, 0.0]
    # ob = np.array([[20.0, 10.0], [30.0, 6.0], [30.0, 8.0], [35.0, 8.0], [50.0, 3.0]])
    way_points_x = [0.0, 40.0]
    way_points_y = [0.0, 40.0]
    ob = np.array([[3.0, 3.0], [10.0, 11.0], [23.0, 25.0]])

    tx, ty, tyaw, tc, csp = generate_target_course(way_points_x, way_points_y)

    # initial state
    s = 0.0  # current course position
    s_d = 10.0 / 3.6  # current speed [m/s]
    s_dd = 0.0  # current acceleration [m/ss]
    d = 2.0  # current lateral position [m]
    d_d = 0.0  # current lateral speed [m/s]
    d_dd = 0.0  # current lateral acceleration [m/s]

    while True:
        path = frenet_optimal_planning(csp, s, s_d, s_dd, d, d_d, d_dd, ob)

        s = path.s[1]
        d = path.d[1]
        d_d = path.d_d[1]
        d_dd = path.d_dd[1]
        s_d = path.s_d[1]
        s_dd = path.s_dd[1]

        if render:
            area = 30
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)
            plt.plot(ob[:, 0], ob[:, 1], "xk")
            plt.plot(path.x[1:], path.y[1:], "-or")
            plt.plot(path.x[1], path.y[1], "vc")
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("v[km/h]:" + str(s_d * 3.6)[0:4])
            plt.grid(True)
            plt.pause(0.0001)

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 2.0:
            print("Goal")
            break

if __name__ == "__main__":
    main()

