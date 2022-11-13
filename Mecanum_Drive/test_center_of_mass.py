#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 19:52:06 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from mecanum_drive import Robot

robot_ideal = Robot(m=100.,I_z=200.)

N = int(1E3)
Gamma = np.zeros((N,4,1))
for i in range(N):
    Gamma[i] = robot_ideal.control_heading_to_torque(np.array([[0.],[0.1],[0.]]))
q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[0],[0],[0]])
robot_ideal.time_integrate(q_r0, q_rdot0, Gamma, N)


robot_real = Robot(m=100.,I_z=200.,x_center=0.1,y_center=0)
q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[0],[0],[0]])
robot_real.time_integrate(q_r0, q_rdot0, Gamma, N)
# robot isn't aware that the calculations aren't accurate to its real center of mass
# uses ideal calculation of Gamma

trajectory = plt.figure()
robot_ideal.plot_trajectory(fig=trajectory, show=False, linecolor='b', label="ideal")
robot_real.plot_trajectory(fig=trajectory, show=False, linecolor='g', label="real")
plt.show()

print("ideal:")
print(robot_ideal.q_r[-1])
print("\nreal:")
print(robot_real.q_r[-1])
print("\ndifference:")
print(robot_real.q_r[-1]-robot_ideal.q_r[-1])