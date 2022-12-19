import numpy as np

from motor_simulation import Motor
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('neverest_with_mecanum_wheel.csv').to_dict('records')

    # motor_consts = []
    # errors = []
    # for motor_const in np.arange(0.3, 0.4, .01):
    #     orbital_20 = Motor(.005, 1.6, motor_const, 0.37)
    #
    #     time = np.zeros(len(df) - 1)
    #     angular_vel_real = np.zeros(len(df) - 1)
    #     angular_vel_pred = np.zeros(len(df) - 1)
    #
    #     for index in range(0, len(df) - 1):
    #         time[index] = df[index]['time']
    #         angular_vel_real[index] = df[index]['motor_velocity']
    #         angular_vel_pred[index] = orbital_20.angular_vel
    #
    #         orbital_20.time_integrate(df[index]['m_power'] * df[index]['batt_voltage'], df[index + 1]['time'] - df[index]['time'])
    #
    #     motor_consts.append(motor_const)
    #     errors.append(np.average((angular_vel_real - angular_vel_pred)))
    #
    # plt.figure()
    # plt.plot(motor_consts, errors)
    # plt.xlabel('motor_const')
    # plt.ylabel('error (rad/s)')
    # plt.title("Simulation error")
    # plt.legend()
    # plt.show()

    orbital_20 = Motor(.005, 1.6, 0.36, 0.37)
    time = np.zeros(len(df) - 1)
    angular_vel_real = np.zeros(len(df) - 1)
    angular_vel_pred = np.zeros(len(df) - 1)

    for index in range(0, len(df) - 1):
        time[index] = df[index]['time']
        angular_vel_real[index] = df[index]['motor_velocity']
        angular_vel_pred[index] = orbital_20.angular_vel

        orbital_20.time_integrate(df[index]['m_power'] * df[index]['batt_voltage'], df[index + 1]['time'] - df[index]['time'])


    plt.figure()
    # plt.plot(time, angular_vel_real - angular_vel_pred, 'g', label="error")
    plt.plot(time, angular_vel_real, 'r', label="Real motor velocity")
    plt.plot(time, angular_vel_pred, 'b', label="Predicted motor velocity")
    plt.xlabel('time')
    plt.title("Real motor speed vs predicted - ramp signals")
    plt.legend()
    plt.show()