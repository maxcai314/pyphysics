import numpy as np
from scipy.signal import lsim2

from motor_simulation import Motor
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('neverest_with_mecanum_wheel.csv').to_dict('records')

    orbital_20 = Motor(.005, 1.6, 0.36, 0.37)
    time = np.array([d['time'] for d in df])
    angular_vel_pred = []
    #
    for index in range(0, len(df) - 1):
        time[index] = df[index]['time']
        angular_vel_pred.append(orbital_20.angular_vel)

        orbital_20.time_integrate(df[index]['m_power'] * df[index]['batt_voltage'],
                                  df[index + 1]['time'] - df[index]['time'])

    angular_vel_real = np.array([d['motor_velocity'] for d in df])
    _, angular_vel_pred_tfn, _ = lsim2(orbital_20.transfer_function, [d['m_power'] * d['batt_voltage'] for d in df], [d['time'] for d in df], hmax=.01)

    angular_vel_pred.insert(0, 0)
    plt.figure()
    # plt.plot(time, angular_vel_real - angular_vel_pred, 'g', label="error")
    plt.plot(time, angular_vel_real, 'r', label="Real motor velocity")
    plt.plot(time, angular_vel_pred, 'b', label="Predicted motor velocity")
    plt.plot(time, angular_vel_pred_tfn, 'g', label="Predicted motor velocity - tfn")
    plt.xlabel('time')
    plt.title("Real motor speed vs predicted - ramp signals")
    plt.legend()
    plt.show()
