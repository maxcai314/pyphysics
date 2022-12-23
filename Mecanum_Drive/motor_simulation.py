#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 10:35:09 2022

@author: maxcai
"""

import numpy as np
from scipy.signal import TransferFunction


class Motor:
    def __init__(self, moment_of_inertia, armature_resistance, motor_constant, viscous_friction, inductance = 0):
        self.moment_of_inertia = moment_of_inertia
        self.armature_resistance = armature_resistance
        self.motor_constant = motor_constant
        self.viscous_friction = viscous_friction

        self.torque = 0
        self.angular_vel = 0
        self.current = 0

        self.inductance = inductance

        self.last_voltage = 0
        self.last_current = 0

        self.transfer_function = TransferFunction(
            [motor_constant],
            [
                inductance * moment_of_inertia,
                armature_resistance * moment_of_inertia + viscous_friction * inductance,
                motor_constant * motor_constant + armature_resistance * viscous_friction
            ]
        )

    # [diff(T(t), t) == (B_m*K_T*e_a(t) + J_m*K_T*diff(E_a(s), s) - (K_T^2 + B_m*R_a)*T(t))/(J_m*R_a)]
    def torque_dot(self, voltage, voltage_dot):
        return (
                (self.viscous_friction * self.motor_constant * voltage) +
                (self.moment_of_inertia * self.motor_constant * voltage_dot) -
                (
                        self.motor_constant * self.motor_constant + self.viscous_friction * self.armature_resistance) * self.torque
        ) / (self.moment_of_inertia * self.armature_resistance)

    # (R_a*F_ext(t) + K_T*e_a(t) - (K_T^2 + B_m*R_a)*diff(theta(t), t))/(J_m*R_a)
    def angular_vel_dot(self, voltage):
        return (
                self.motor_constant * voltage -
                (self.motor_constant ** 2 + self.viscous_friction * self.armature_resistance) * self.angular_vel
        ) / (self.moment_of_inertia * self.armature_resistance)

    # (B_m*e_a(t) - (K_T^2 + B_m*R_a)*i_a(t) + J_m*diff(e_a(t), t))/(J_m*R_a)
    def current_dot(self, voltage, time_step):
        return (
                self.viscous_friction * voltage -
                (self.motor_constant ** 2 + self.viscous_friction * self.armature_resistance) * self.current +
                self.moment_of_inertia * (voltage - self.last_voltage) / time_step
        )


    def time_integrate_(self, voltage, time_step, external_force):
        voltage += self.armature_resistance / self.motor_constant * external_force # backdrive
        torque_dot = self.torque_dot(voltage, (voltage - self.last_voltage) / time_step)
        angular_vel_dot = self.angular_vel_dot(voltage)
        current_dot = self.current_dot(voltage, time_step)

        self.torque += torque_dot * time_step
        self.angular_vel += angular_vel_dot * time_step
        self.current += current_dot * time_step

        self.last_voltage = voltage

    def time_integrate(self, voltage, time, time_step=.01, external_force=0):
        time = max(time, time_step)
        # an np array of [time, voltage, current, angular_vel, torque]
        data = np.zeros((int(time / time_step), 5))
        for i in range(int(time / time_step)):
            self.time_integrate_(voltage, time_step, external_force)
            data[i] = [i * time_step, voltage, self.current, self.angular_vel, self.torque]
        return data

    def time_integrate_multiple(self, voltages, times, time_step=.01, external_force=0):
        # for each voltage, time pair, integrate the motor until the next voltage, time pair
        data = np.zeros((len(voltages), 5))
        data[0] = [0, voltages[0], self.current, self.angular_vel, self.torque]
        for i in range(1, len(voltages)):
            result = self.time_integrate(voltages[i - 1], times[i] - times[i - 1], time_step, external_force)
            data[i] = result[-1]
        return data

    def get_state(self):
        return np.array([[self.angular_vel], [self.torque]])

    def to_string(self):
        return f'Angular Velocity {self.angular_vel}, Torque {self.torque}'

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    motor = Motor(.01, 1.6, .366, 0)

    total_time = .5
    dt = 0.005
    N = int(total_time / dt)

    simulation_time = np.arange(0, N) * dt
    state = np.zeros((N, 2, 1))
    for i in range(0, N):
        state[i] = motor.get_state()
        motor.time_integrate(6, dt)
        print(motor.to_string())

    # for i in range(50000,N):
    #     state[i] = motor.get_state()
    #     motor.time_integrate(0, dt)

    plt.figure()
    angular_vel = state[:, 0, 0]
    torque = state[:, 1, 0]
    plt.plot(simulation_time, angular_vel, 'r', label="angular velocity")
    plt.plot(simulation_time, torque, 'b', label="torque")
    plt.xlabel('time')
    plt.ylabel('state')
    plt.title("Motor Test")
    plt.legend()
    plt.show()
