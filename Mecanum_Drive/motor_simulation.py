#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 10:35:09 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import floor


class Motor:
    def __init__(self, moment_of_inertia, armature_resistance, motor_constant, viscous_friction):
        self.moment_of_inertia = moment_of_inertia
        self.armature_resistance = armature_resistance
        self.motor_constant = motor_constant
        self.viscous_friction = viscous_friction

        self.torque = 0
        self.angular_vel = 0
        self.last_voltage = 0

    # [diff(T(t), t) == (B_m*K_T*e_a(t) + J_m*K_T*diff(E_a(s), s) - (K_T^2 + B_m*R_a)*T(t))/(J_m*R_a)]
    def torque_dot(self, voltage, voltage_dot):
        return (
                (self.viscous_friction * self.motor_constant * voltage) +
                (self.moment_of_inertia * self.motor_constant * voltage_dot) -
                (self.motor_constant * self.motor_constant + self.viscous_friction * self.armature_resistance) * self.torque
        ) / (self.moment_of_inertia * self.armature_resistance)

    # (K_T*E_a(t) - (K_T^2 + B_m*R_a)*O(t))/(J_m*R_a)
    def angular_vel_dot(self, voltage):
        return (
                self.motor_constant * voltage -
                (self.motor_constant ** 2 + self.viscous_friction * self.armature_resistance) * self.angular_vel
        ) / (self.moment_of_inertia * self.armature_resistance)

    def time_integrate_(self, voltage, time_step):
        torque_dot = self.torque_dot(voltage, (voltage - self.last_voltage) / time_step)
        angular_vel_dot = self.angular_vel_dot(voltage)

        self.torque += torque_dot * time_step
        self.angular_vel += angular_vel_dot * time_step
        self.last_voltage = voltage
    def time_integrate(self, voltage, time_step):
        if time_step < .01:
            self.time_integrate_(voltage, time_step)
        else:
            num_steps = time_step / .01
            curr_time = 0
            for i in range(int(num_steps)):
                self.time_integrate_(voltage, .01)
                curr_time += .01

            self.time_integrate_(voltage, time_step - curr_time)


    def get_state(self):
        return np.array([[self.angular_vel], [self.torque]])

    def to_string(self):
        return f'Angular Velocity {self.angular_vel}, Torque {self.torque}'


if __name__ == "__main__":
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
