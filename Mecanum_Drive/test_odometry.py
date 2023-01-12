#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:12:30 2022

@author: maxcai
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mecanum_drive import Robot

q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[1],[1],[2]])

robot = Robot(q_r=q_r0, q_rdot=q_rdot0)

N = int(1E3)
Gamma = np.zeros((N,4,1))
for i in range(N):
    Gamma[i] = robot.control_heading_to_torque(np.array([[0.075],[0.],[-0.0714285714]]))

robot.time_integrate_steps(Gamma, N)

robot.simulate_odometry_from_list()

coarseness = 100
t_coarse = robot.t_list[0::coarseness]
q_d_coarse = robot.q_d_list[0::coarseness]
# q_r_predict = robot.predict_position_from_odometry(q_d_coarse)

def odometry_curve(t,t_coarse,q_d_coarse):
    q_d_interp = np.zeros((3,1))
    for i in range(3):
        q_d_interp[i] = np.interp(t,t_coarse,q_d_coarse[:,i,0])
    return q_d_interp

odometry_function_interp = interpolate.interp1d(t_coarse,q_d_coarse,kind="cubic",axis=0,fill_value="extrapolate")
odometry_function_nointerp = interpolate.interp1d(t_coarse,q_d_coarse,kind="nearest",axis=0,fill_value="extrapolate")
dt_integrate = 5E-4
t_integrate = np.arange(0, robot.t_list[-1], dt_integrate)
q_r_predict_interp = robot.predict_position_from_odometry(t_integrate,odometry_function_interp,x0=q_r0)
q_r_predict_nointerp = robot.predict_position_from_odometry(t_integrate,odometry_function_nointerp)

N_interp = 100
t_interp = np.arange(N_interp)/N_interp * 5
q_d_interp = odometry_function_interp(t_interp)
q_d_nointerp = odometry_function_nointerp(t_interp)
# q_d_interp = np.zeros((N_interp,3,1))
# for i in range(N_interp):
    # q_d_interp[i] = odometry_curve(t_interp[i],t_coarse,q_d_coarse)

final_error = robot.q_r_list[-1] - q_r_predict_interp[-1]
print('Error between estimates')
print(final_error)

fig1 = plt.figure(1)
plt.plot(robot.t_list,robot.q_d_list[:,0,:],'r',label = 'Left Wheel')
plt.plot(robot.t_list,robot.q_d_list[:,1,:],'b',label = 'Right Wheel')
plt.plot(robot.t_list,robot.q_d_list[:,2,:],'g',label = 'Back Wheel')
plt.xlabel('t')
plt.ylabel('odometry wheel angle')
plt.title('Odometry')
plt.legend()
plt.show()

fig2 = plt.figure(2)
# maxd = np.max(np.abs(np.array([robot.q_r_list[:,0,0],robot.q_r[:,1,0]])))
# plt.xlim(-1.1*maxd,1.1 * maxd)
# plt.ylim(-1.1*maxd,1.1 * maxd)
plt.xlim(-0.5,3.5)
plt.ylim(-1.5,1.75)
plt.plot(q_r_predict_nointerp[:,0,0],q_r_predict_nointerp[:,1,0],'m',linewidth=4,label='uninterpolated')
plt.plot(q_r_predict_interp[:,0,0],q_r_predict_interp[:,1,0],'r',linewidth=4,label='interpolated')
plt.plot(robot.q_r_list[:,0,0],robot.q_r_list[:,1,0],'b',label='actual motion')
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
