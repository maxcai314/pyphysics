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
Gamma_real = Gamma
# robot isn't aware that the calculations aren't accurate to its real center of mass
# uses ideal calculation of Gamma
q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[0],[0],[0]])
robot_real.time_integrate(q_r0, q_rdot0, Gamma_real, N)

robot_real_adjusted = Robot(m=100.,I_z=200.,x_center=0.1,y_center=0)
Gamma_adjusted = np.zeros((N,4,1))
for i in range(N):
    Gamma_adjusted[i] = robot_real_adjusted.control_heading_to_torque(np.array([[0.],[0.1],[0.]]))
    # uses this robot's dimensions to calculate what should be the ideal motor powers
    # doesn't seem to actually fix the issue, for whatever reason
    # maybe this issue is with the physics model itself?
q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[0],[0],[0]])
robot_real_adjusted.time_integrate(q_r0, q_rdot0, Gamma_adjusted, N)
# robot uses adjusted calculations, should be accurate
# but isnt??

trajectory = plt.figure()
robot_ideal.plot_trajectory(fig=trajectory, show=False, linecolor='b', label="ideal")
robot_real.plot_trajectory(fig=trajectory, show=False, linecolor='g', label="real")
robot_real_adjusted.plot_trajectory(fig=trajectory, show=False, linecolor='m', label="real improved")
plt.show()

print("\nideal:")
print(robot_ideal.q_r[-1])
print("\nreal:")
print(robot_real.q_r[-1])
print("\nreal improved:")
print(robot_real_adjusted.q_r[-1])
print("\ndifference:")
print(robot_real.q_r[-1]-robot_ideal.q_r[-1])
print("\ndifference with improvement:")
print(robot_real_adjusted.q_r[-1]-robot_ideal.q_r[-1])