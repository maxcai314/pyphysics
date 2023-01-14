#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain

startPos = np.array([[0],[0],[0]]) # x, y, angle
startVel = np.array([[0],[0],[0]]) # xVel, yVel, angleVel

robot = Drivetrain(voltage=12, startPos=startPos, startVel=startVel)

N = int(1E4)
time_step = 1E-3
t = np.arange(0, N*time_step, time_step)

robot_position = np.zeros((N,3,1))
robot_velocity = np.zeros((N,3,1))
robot_position[0] = startPos
robot_velocity[0] = startVel

for i in range(1,2000):
    robot.set_powers(1, 1, 1, 1)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity
    
for i in range(2000, 4000):
    robot.set_powers(0, 0, 0, 0)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity
    
for i in range(4000, 6000):
    robot.set_powers(-1, -1, -1, -1)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity

for i in range(6000, N):
    robot.set_powers(0, 0, 0, 0)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity

fig1, axis = plt.subplots(2)
fig2=plt.figure()

robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1, show=False)
# robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1, show=True)
robot.plot_trajectory(robot_position, fig=fig2)