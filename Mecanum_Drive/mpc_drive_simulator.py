#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
from casadi import *
# import numpy as np
import aerosandbox.numpy as np


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

    def torque(self, position, velocity, inputs):
        angle = position[2]
        voltage = inputs[4]

        rotation = rotationmatrix(angle)
        ea = voltage * inputs[:4]
        eb = self.R @ rotation.T @ velocity * self.motor_constant
        return (ea - eb) / self.armature_resistance

    def continuous_dynamics(self, state, inputs):
        position = state[:3]
        velocity = state[3:]

        acceleration = self.get_aceleration(position, velocity, self.torque(position, velocity, inputs))
        return vertcat(velocity, acceleration)


if __name__ == '__main__':
    robot = DriveModel()
    position = vertcat(0, 0, 0)  # x, y, angle (meters, meters, radians)
    velocity = vertcat(0, 0, 0)  # units/second
    powers = vertcat(1, 1, 1, 1)  # powers must be between -1, 1 for each motor
    voltage = 12  # battery voltage

    state = vertcat(position, velocity)
    input = vertcat(powers, voltage)

    print(vertsplit(robot.continuous_dynamics(state, input)))
