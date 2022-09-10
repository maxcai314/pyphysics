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

robot = Robot()

N = int(1E3)
Gamma = np.zeros((N,4,1))
for i in range(50):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[1]]))
for i in range(50,150):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[-1]]))
for i in range(150,200):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[1]]))
for i in range(200,350):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[-1]]))
for i in range(350,400):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[1]]))
for i in range(400,450):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[-2]]))
for i in range(450,500):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[1]]))
for i in range(500,550):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[-1]]))
for i in range(550,600):
    Gamma[i] = robot.control_heading_to_torque(np.array([[1],[0],[-1]]))

q_r0 = np.array([[0],[-2],[0]])
q_rdot0 = np.array([[0],[0],[0]])

robot.time_integrate(q_r0, q_rdot0, Gamma, N)

robot.simulate_odometry()

#updates at 100hz, coarseness = 100 is 1 update per second
coarseness = 10
t_coarse = robot.t[0::coarseness]
q_d_coarse = robot.q_d[0::coarseness]
np.random.seed(10)
standard_deviation_radians = 0.05
q_d_coarse_noise = np.random.normal(np.zeros((3,1)),standard_deviation_radians,q_d_coarse.shape)
q_d_coarse += q_d_coarse_noise

def odometry_curve(t,t_coarse,q_d_coarse):
    q_d_interp = np.zeros((3,1))
    for i in range(3):
        q_d_interp[i] = np.interp(t,t_coarse,q_d_coarse[:,i,0])
    return q_d_interp

odometry_function_cubic = interpolate.interp1d(t_coarse,q_d_coarse,kind="cubic",axis=0,fill_value="extrapolate")
odometry_function_regular = interpolate.interp1d(t_coarse,q_d_coarse,kind="nearest",axis=0,fill_value="extrapolate")
dt_integrate = 5E-4
t_integrate = np.arange(0, robot.t[-1], dt_integrate)
q_r_predict_cubic = robot.predict_position_from_odometry(t_integrate,odometry_function_cubic,x0=q_r0)
q_r_predict_regular = robot.predict_position_from_odometry(t_integrate,odometry_function_regular,x0=q_r0)

N_interp = 1000
t_interp = np.arange(N_interp)/N_interp * 5
q_d_interp = odometry_function_cubic(t_interp)
# q_d_interp = np.zeros((N_interp,3,1))
# for i in range(N_interp):
    # q_d_interp[i] = odometry_curve(t_interp[i],t_coarse,q_d_coarse)

final_error = robot.q_r[-1] - q_r_predict_cubic[-1]
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
plt.plot(robot.q_r[:,0,0],robot.q_r[:,1,0],'b',label='actual')
plt.plot(q_r_predict_regular[:,0,0],q_r_predict_regular[:,1,0],'g',label='predict raw data',alpha=0.4)
plt.plot(q_r_predict_cubic[:,0,0],q_r_predict_cubic[:,1,0],'r',label='predict interpolated',alpha=0.4)
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
