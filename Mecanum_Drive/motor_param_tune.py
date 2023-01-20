import numpy as np
from scipy.signal import lsim2

from motor_simulation import Motor
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('macosx')

import multiprocessing as mp

df = pd.read_csv('neverest_with_mecanum_wheel.csv')
angular_vel_real = df['motor_velocity']
voltages = df['m_power'] * df['batt_voltage']
times = df['time'].to_numpy()


def simulate(*args):
    orbital_20 = Motor(*args)
    angular_vel_pred = orbital_20.time_integrate_multiple(voltages, times)[:, 3]
    # print(angular_vel_pred)
    return np.average((angular_vel_real - angular_vel_pred)[df['time'] < 1.5] ** 2), angular_vel_pred


def simulate_nosave(*args):
    orbital_20 = Motor(*args)
    angular_vel_pred = orbital_20.time_integrate_multiple(voltages, times)[:, 3]
    return np.average(abs(angular_vel_real - angular_vel_pred))


if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:

        def gradient_mp(a, b, c, d):
            result = pool.starmap(simulate_nosave,
                                  [(a + .000001, b, c, d), (a - .000001, b, c, d), (a, b + .0001, c, d),
                                   (a, b - .0001, c, d), (a, b, c + .0001, d), (a, b, c - .0001, d),
                                   (a, b, c, d + .0001), (a, b, c, d - .0001)])
            de_da = (result[0] - result[1]) / .000002
            de_db = (result[2] - result[3]) / .0002
            de_dc = (result[4] - result[5]) / .0002
            de_dd = (result[6] - result[7]) / .0002

            return np.array([de_da * 1E-5, de_db * 1E-2, de_dc * 1E-3, de_dd * 1E-3])


        start = np.array([.01, 1.6, 0.105195862, 0.0])
        current = np.array(start)

        # live-updating plot
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlim(0, times[-1])
        real, = ax.plot(times, angular_vel_real, 'r', label="angular velocity")
        predicted, = ax.plot(times, np.zeros_like(times), 'b', label="predicted angular velocity")
        plt.legend()
        plt.show()

        for i in range(1000):
            error, predicted_values = simulate(*current)
            grad = gradient_mp(*current)
            grad[1] = 0
            grad[2] = 0
            print(f"{i}:  {repr(current)}, error: {error}, grad = {grad}")
            current -= grad * 1E-3
            predicted.set_ydata(predicted_values)
            fig.canvas.draw()
            fig.canvas.flush_events()
