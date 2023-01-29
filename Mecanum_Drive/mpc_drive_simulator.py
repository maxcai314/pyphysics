#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
import casadi
# import numpy as np
import aerosandbox.numpy as np

def rotationmatrix(self, psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

def rotationmatrixdot(self, psi, psidot):
    return np.array([[-np.sin(psi), -np.cos(psi), 0], [np.cos(psi), -np.sin(psi), 0], [0, 0, 0]]) * psidot

class DriveModel():
    def __init__(self, motor_constant=0.22,I_z=3.,I_w=[0.05,0.05,0.05,0.05],L=0.2,l=0.15,m=11.2,r=0.048,friction=0.1, voltage = 12, q_r=None, q_rdot=None):
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

    def get_aceleration(self, position, torques):
        angle = position[2]
        rotation = rotationmatrix(angle)
        rotationdot = rotationmatrixdot(angle, angle)

        q_wdot = self.R @ np.linalg.inv(rotation) @ angle

        H = self.M_r + rotation @ self.R.T @ self.M_w @ self.R @ np.linalg.inv(rotation)
        K = rotation @ self.R.T @ self.M_w @ self.R @ rotationdot.T
        F_a = rotation @ (self.R.T @ (torques - np.sign(q_wdot) * self.friction))
        q_rddot = np.linalg.inv(H) @ (F_a - K @ torques)
        return q_rddot

    def torque(self, position, velocity, inputs):
        angle = position[2]

        rotation = self.rotationmatrix(angle)
        ea = self.voltage * inputs[:4]
        eb = self.R @ rotation.T @ velocity * self.motor_constant

        return (ea - eb) / self.armature_resistance

    def continuous_dynamics(self, state, inputs):
        position = state[:3]
        velocity = state[3:]

        self.q_rddot = self.get_aceleration(position, self.torque(position, velocity, inputs))
        return; #put stuff here lol




