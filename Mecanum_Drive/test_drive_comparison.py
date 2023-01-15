#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain
import pandas as pd
from scipy import interpolate



df = pd.read_csv('driving_around_log7.csv').to_dict('records')
x_pos_real = np.array([d['x_position'] for d in df]) / 39.37 # convert from inch to meter
y_pos_real = np.array([d['y_position'] for d in df]) / 39.37
angle_real = np.array([d['angle'] for d in df])
x_vel_real = np.array([d['x_velocity'] for d in df]) / 39.37
y_vel_real = np.array([d['y_velocity'] for d in df]) / 39.37
angle_vel_real = np.array([d['angular_velocity'] for d in df])
fl = np.array([d['fl'] for d in df])
fr = np.array([d['fr'] for d in df])
bl = np.array([d['bl'] for d in df])
br = np.array([d['br'] for d in df])

voltage_real = [d['battery_voltage'] for d in df]
times_real = np.array([d['time'] for d in df])

voltage_function = interpolate.interp1d(times_real, voltage_real, kind="cubic", fill_value="extrapolate")

flPower = interpolate.interp1d(times_real, fl, kind="cubic", fill_value="extrapolate")
frPower = interpolate.interp1d(times_real, fr, kind="cubic", fill_value="extrapolate")
blPower = interpolate.interp1d(times_real, bl, kind="cubic", fill_value="extrapolate")
brPower = interpolate.interp1d(times_real, br, kind="cubic", fill_value="extrapolate")

x_vel =interpolate.interp1d(times_real, x_vel_real, kind="linear", fill_value="extrapolate")
y_vel =interpolate.interp1d(times_real, y_vel_real, kind="linear", fill_value="extrapolate")
angle_vel =interpolate.interp1d(times_real, angle_vel_real, kind="linear", fill_value="extrapolate")

startPos = np.array([[0],[0],[0]]) # x, y, angle
startVel = np.array([[0],[0],[0]]) # xVel, yVel, angleVel

robot = Drivetrain(voltage=12, startPos=startPos, startVel=startVel)

N = int(14E4)
time_step = 1E-4
t = np.arange(0, N*time_step, time_step)

x_accel_real = np.zeros((N,1))
y_accel_real = np.zeros((N,1))
angle_accel_real = np.zeros((N,1))
derivative_step = 0.1
for i in range(N):
    currentTime = t[i]
    
    x_accel_real[i] = (x_vel(currentTime + derivative_step) - x_vel(currentTime))/derivative_step
    y_accel_real[i] = (y_vel(currentTime + derivative_step) - y_vel(currentTime))/derivative_step
    angle_accel_real[i] = (angle_vel(currentTime + derivative_step) - angle_vel(currentTime))/derivative_step


robot_position = np.zeros((N,3,1))
robot_velocity = np.zeros((N,3,1))
robot_acceleration = np.zeros((N,3,1))
robot_position[0] = startPos
robot_velocity[0] = startVel

for i in range(N):
    currentTime = t[i]
    robot.voltage = voltage_function(currentTime)
    robot.set_powers(flPower(currentTime),frPower(currentTime),blPower(currentTime),brPower(currentTime))
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity
    robot_acceleration[i] = robot.acceleration

robot_position[:,2,0] = ((robot_position[:,2,0]+np.pi)%(2*np.pi)) - np.pi # angle wrap

fig1, axis = plt.subplots(2)
fig2 = plt.figure()

ax1 = fig1.axes[0]
ax2 = fig1.axes[1]

# ax1.plot(times_real, x_pos_real,'c', label='robot X')
# ax1.plot(times_real, y_pos_real,'y', label='robot Y')
ax1.plot(times_real, angle_real,'m', label='robot Psi')

# ax2.plot(times_real, x_vel_real,'c', label='robot X vel')
# ax2.plot(times_real, y_vel_real,'y', label='robot Y vel')
ax2.plot(times_real, angle_vel_real,'m', label='robot Psi vel')

robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1, show=True)
# robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1, show=True)
robot.plot_trajectory(robot_position, fig=fig2)

plt.figure()

plt.plot([0, np.max(t)],[0,0],'k')
# plt.plot(t, x_accel_real,'c', label='X accel real')
# plt.plot(t, y_accel_real,'y', label='Y accel real')
plt.plot(t, angle_accel_real,'m', label='Psi accel real')
# plt.plot(t, robot_acceleration[:,0,0],'b', label='X accel simulate')
# plt.plot(t, robot_acceleration[:,1,0],'r', label='Y accel simulate')
plt.plot(t, robot_acceleration[:,2,0],'g', label='Psi accel simulate')

plt.legend()
plt.title("Acceleration")
plt.show()