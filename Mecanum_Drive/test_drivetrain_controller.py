#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:44:15 2023

@author: maxcai
"""
import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain

startPos = np.array([[0],[0],[0]]) # x, y, angle
startVel = np.array([[0],[0],[0]]) # xVel, yVel, angleVel

robot = Drivetrain(startPos=startPos, startVel=startVel)

N = int(1E4)
time_step = 1E-3
t = np.arange(0, N*time_step, time_step)

robot_position = np.zeros((N,3,1))
robot_velocity = np.zeros((N,3,1))
robot_position[0] = startPos
robot_velocity[0] = startVel

for i in range(1,4000):
    robot.set_powers_with_controller(2**0.5, 2**0.5, 1, scaleDown = True)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity
    
for i in range(4000, 5000):
    robot.set_powers_with_controller(1, 0, 0, scaleDown = True)
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity
    
for i in range(5000, N):
    robot.set_powers_with_controller(2**0.5, 2**0.5, 1, scaleDown = False) # without scaledown, the path is much weirder
    robot.time_integrate(time_step)
    
    robot_position[i] = robot.position
    robot_velocity[i] = robot.velocity

fig1, axis = plt.subplots(2)
fig2=plt.figure()

robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1)
robot.plot_trajectory(robot_position, fig=fig2)