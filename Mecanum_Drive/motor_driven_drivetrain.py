#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:11:23 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from mecanum_drive import Robot
from motor_simulation import Motor

robot = Robot(m=10.,I_z=0.15,I_w=[0.005,0.005,0.005,0.005], friction=0.059, r=0.048)

front_left = Motor(.005, 1.6, 0.36, 0.37)
front_right = Motor(.005, 1.6, 0.36, 0.37)
back_left = Motor(.005, 1.6, 0.36, 0.37)
back_right = Motor(.005, 1.6, 0.36, 0.37)

q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[1],[1],[2]])

N = int(1E3)
dt = 1E-2
t = np.arange(0, N*dt, dt)

Gamma = np.zeros((4,1))

q_r = np.zeros((N,3,1))
q_rdot = np.zeros((N,3,1))
q_r[0] = q_r0
q_rdot[0] = q_rdot0

vFrontLeft = 12
vFrontRight = 12
vBackLeft = 12
vBackRight = 12

for i in range(1,N):
    if i>500:
        vFrontLeft = 0
        vFrontRight = 0
        vBackLeft = 0
        vBackRight = 0
    
    front_left.time_integrate(vFrontLeft, dt)
    front_right.time_integrate(vFrontRight, dt)
    back_left.time_integrate(vBackLeft, dt)
    back_right.time_integrate(vBackRight, dt)

    Gamma[0,0] = front_left.torque
    Gamma[1,0] = front_right.torque
    Gamma[2,0] = back_left.torque
    Gamma[3,0] = back_right.torque
    
    robot.time_integrate(Gamma, dt=dt)
    q_r[i] = robot.q_r
    q_rdot[i] = robot.q_rdot

fig1, axis = plt.subplots(2)
fig2=plt.figure()

robot.plot_evolution(t, q_r, q_rdot, fig=fig1)
robot.plot_trajectory(q_r, fig=fig2)