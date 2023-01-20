import numpy as np
from scipy.signal import lsim2

from motor_simulation import Motor
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('neverest_with_mecanum_wheel.csv')

    orbital_20 = Motor(0.00036288, 1., 0.337033998, 0.0025)
    time = df['time']
    angular_vel_pred = []
    df['motor_voltage'] = df['m_power'] * df['batt_voltage']
    df['time_diff'] = df['time'].diff()
    for i, row in df.iterrows():
        if i == 0:
            continue
        angular_vel_pred.append(orbital_20.angular_vel)

        print(row)
        orbital_20.time_integrate(row['motor_voltage'],
                                  row['time_diff'])

    angular_vel_real = df['motor_velocity']
    _, angular_vel_pred_tfn, _ = lsim2(orbital_20.transfer_function, df['motor_voltage'], df['time'], hmax=.01)

    angular_vel_pred.insert(0, 0)
    plt.figure()
    # plt.plot(time, angular_vel_real - angular_vel_pred, 'g', label="error")
    plt.plot(time, angular_vel_real, 'r', label="Real motor velocity")
    plt.plot(time, angular_vel_pred, 'b', label="Predicted motor velocity")
    # plt.plot(time, angular_vel_pred_tfn, 'g', label="Predicted motor velocity - tfn")
    plt.xlabel('time')
    plt.title("Real motor speed vs predicted - ramp signals")
    plt.legend()
    plt.show()
