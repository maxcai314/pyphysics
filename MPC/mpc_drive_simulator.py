#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
from casadi import *
# import numpy as np
import aerosandbox.numpy as np

from Mecanum_Drive.mecanum_data import DataSeries


def rotationmatrix(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])


def rotationmatrixdot(psi, psidot):
    return np.array([[-np.sin(psi), -np.cos(psi), 0], [np.cos(psi), -np.sin(psi), 0], [0, 0, 0]]) * psidot


class DriveModel():
    def __init__(self, motor_constant=0.36827043, I_z=1.55687398, I_w=np.ones(4) * 0.0549806, L=0.115, l=0.1325, m=11.2,
                 r=0.048, friction=np.array([[0.05758405], [0.00923156], [0.06716439], [0.02307628]])):
        self.armature_resistance = 1.6
        self.motor_constant = motor_constant

        self.L = L
        self.l = l
        self.r = r
        self.m = m
        self.I_z = I_z
        self.I_w = I_w
        self.friction = friction

        self.M_r = np.diag([self.m, self.m, self.I_z])
        self.M_w = np.diag(self.I_w)

        self.S = np.array([self.L, self.L, -self.L, -self.L])
        self.d = np.array([self.l, -self.l, self.l, -self.l])
        self.alpha = [0.25 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.25 * np.pi]

        self.R = np.zeros((4, 3))
        for i in range(4):
            self.R[i] = 1 / self.r * np.array(
                [1, -np.tan(self.alpha[i]) ** -1, -self.d[i] - self.S[i] * (np.tan(self.alpha[i]) ** -1)])

    def get_aceleration(self, position, velocity, torques):
        angle = position[2]
        angleVel = velocity[2]
        rotation = rotationmatrix(angle)
        rotationdot = rotationmatrixdot(angle, angleVel)

        q_wdot = self.R @ np.linalg.inv(rotation) @ velocity

        H = self.M_r + rotation @ self.R.T @ self.M_w @ self.R @ np.linalg.inv(rotation)
        K = rotation @ self.R.T @ self.M_w @ self.R @ rotationdot.T
        F_a = rotation @ (self.R.T @ (torques - np.sign(q_wdot) * self.friction))
        q_rddot = np.linalg.inv(H) @ (F_a - K @ velocity)
        return q_rddot

    def torque(self, position, velocity, voltage, inputs):
        angle = position[2]
        powers = inputs[:4]

        rotation = rotationmatrix(angle)
        ea = voltage * powers
        eb = self.R @ rotation.T @ velocity * self.motor_constant
        return (ea - eb) / self.armature_resistance

    def continuous_dynamics(self, state, inputs, parameters):
        position = state[:3]
        velocity = state[3:]
        voltage = parameters[0]

        acceleration = self.get_aceleration(position, velocity, self.torque(position, velocity, voltage, inputs))
        return vertcat(velocity, acceleration)

    def eval_obj(self, z, p):  # made non-static just so it's easier to call
        # z = z.reshape((-1,))
        # p = p.reshape((-1,))
        [desired_values, weights] = horzsplit(p[1:].reshape((2, -1)).T)
        result = weights * (z - desired_values)
        return np.sum(result**2)

def get_configurable_parameters(
        target_position=vertcat(0, 0, 0),
        target_velocity=vertcat(0, 0, 0),

        motor_voltage_weights=vertcat(0, 0, 0, 0),
        position_weights=vertcat(1, 1, 1),
        velocity_weights=vertcat(0, 0, 0),
):
    configurable_parameters = np.zeros((10, 2))
    configurable_parameters[:4, 1] = motor_voltage_weights.T
    configurable_parameters[4:7, 0] = target_position.T
    configurable_parameters[4:7, 1] = position_weights.T
    configurable_parameters[7:10, 0] = target_velocity.T
    configurable_parameters[7:10, 1] = velocity_weights.T
    return configurable_parameters.reshape((-1,))

if __name__ == '__main__':
    robot = DriveModel()
    position = vertcat(0, 0, 0)  # x, y, angle (meters, meters, radians)
    velocity = vertcat(0, 0, 0)  # units/second

    powers = vertcat(1, 1, 1, 1)  # powers must be between -1, 1 for each motor

    state = vertcat(position, velocity)


    data = DataSeries.from_csv("../Mecanum_Drive/drive_samples/driving_around_log_slower_8.csv")
    configurable_parameters = get_configurable_parameters()
    all_configurable_params = np.zeros((len(data), len(configurable_parameters)))
    all_configurable_params[:] = configurable_parameters
    all_params = np.hstack((data.battery_voltage.reshape((-1,1)), all_configurable_params))

    position_states = np.zeros((len(data), 3))

    for i in range(len(data)):
        input = vertcat(data.fl[i], data.fr[i], data.bl[i], data.br[i])
        state += robot.continuous_dynamics(state, input, all_params[i]) * data.T
        position_states[i] = state[:3].T

    import matplotlib.pyplot as plt

    plt.xlabel("time")
    plt.ylabel("x position")
    plt.plot(data.time, position_states[:, 2], label="x position pred")
    plt.plot(data.time, data.angle, label="x position actual")
    plt.show()
