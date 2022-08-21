#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:12:30 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from mecanum_drive import Robot

robot = Robot()

N = int(1E3)
Gamma = np.zeros((N,4,1))
for i in range(N):
    Gamma[i] = robot.control_heading_to_torque(np.array([[0.075],[0.],[-0.0714285714]]))
q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[1],[1],[2]])

robot.time_integrate(q_r0, q_rdot0, Gamma, N)

robot.simulate_odometry()

coarseness = 20
t_coarse = robot.t[0::coarseness]
q_d_coarse = robot.q_d[0::coarseness]
# q_r_predict = robot.predict_position_from_odometry(q_d_coarse)

def odometry_curve(t,t_coarse,q_d_coarse):
    q_d_interp = np.zeros((3,1))
    for i in range(3):
        q_d_interp[i] = np.interp(t,t_coarse,q_d_coarse[:,i,0])
    return q_d_interp

q_r_predict = robot.predict_position_from_odometry(robot.t,odometry_curve,t_coarse,q_d_coarse)

# N_interp = 100
# t_interp = np.arange(N_interp)/N_interp * 5
# q_d_interp = np.zeros((N_interp,3,1))
# for i in range(N_interp):
#     q_d_interp[i] = odometry_curve(t_interp[i],t_coarse,q_d_coarse)

final_error = robot.q_r[-1] - q_r_predict[-1]
print('Error between estimates')
print(final_error)

fig1 = plt.figure(1)
plt.plot(robot.t,robot.q_d[:,0,:],'r',label = 'Left Wheel')
plt.plot(robot.t,robot.q_d[:,1,:],'b',label = 'Right Wheel')
plt.plot(robot.t,robot.q_d[:,2,:],'g',label = 'Back Wheel')
plt.xlabel('t')
plt.ylabel('odometry wheel angle')
plt.title('Odometry')
plt.legend()
plt.show()

fig2 = plt.figure(2)
maxd = np.max(np.abs(np.array([robot.q_r[:,0,0],robot.q_r[:,1,0]])))
plt.xlim(-1.1*maxd,1.1 * maxd)
plt.ylim(-1.1*maxd,1.1 * maxd)
plt.plot(q_r_predict[:,0,0],q_r_predict[:,1,0],'r',linewidth=4,label='predict')
plt.plot(robot.q_r[:,0,0],robot.q_r[:,1,0],'b',label='actual')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Compare Trajectory')
plt.legend()
plt.gca().set_aspect('equal')
plt.show()

# fig3 = plt.figure(3)
# plt.plot(t_interp,q_d_interp[:,0,:],'r',label = 'Left Wheel')
# plt.plot(t_interp,q_d_interp[:,1,:],'b',label = 'Right Wheel')
# plt.plot(t_interp,q_d_interp[:,2,:],'g',label = 'Back Wheel')
# plt.xlabel('t')
# plt.ylabel('odometry wheel angle')
# plt.title('Odometry Interpolated')
# plt.legend()
# plt.show()
