#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 18:11:23 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from mecanum_drive import Robot

class Drivetrain(Robot):
    
    def __init__(self,motor_constant=0.22,I_z=3.,I_w=[0.05,0.05,0.05,0.05],friction=0.1, static_friction=0, voltage=12, startPos=None, startVel=None):
        super().__init__(L=0.115,l=0.1325,m=11.2,I_z=I_z,I_w=np.ones(4) * I_w, friction=friction, static_friction=static_friction, r=0.048, q_r=startPos, q_rdot=startVel)
        
        self.voltage = voltage
    
        
        self.front_left_power = 0
        self.front_right_power = 0
        self.back_left_power = 0
        self.back_right_power = 0
        
        self.armature_resistance = 1.6
        self.motor_constant  = motor_constant
    
    def range_function(self, input):
        return np.clip(input, -1, 1)
    
    def set_powers(self, front_left_power, front_right_power, back_left_power, back_right_power):
        front_left_power = self.range_function(front_left_power)
        front_right_power = self.range_function(front_right_power)
        back_left_power = self.range_function(back_left_power)
        back_right_power = self.range_function(back_right_power)
        
        self.front_left_power = front_left_power
        self.front_right_power = front_right_power
        self.back_left_power = back_left_power
        self.back_right_power = back_right_power
    
    def set_powers_with_controller(self, forwardMov, strafeMov, turnMov, scaleDown = True):
        front_left_power = forwardMov + strafeMov - turnMov
        front_right_power = forwardMov - strafeMov + turnMov
        back_left_power = forwardMov - strafeMov - turnMov
        back_right_power = forwardMov + strafeMov + turnMov
        
        maximumPower = np.max(np.abs([front_left_power, front_right_power, back_left_power, back_right_power]))
        if maximumPower > 1:
            if scaleDown:
                front_left_power /= maximumPower
                front_right_power /= maximumPower
                back_left_power /= maximumPower
                back_right_power /= maximumPower
        
        self.set_powers(front_left_power, front_right_power, back_left_power, back_right_power)
    
    def torque(self):
        Rotation = self.rotationmatrix(self.q_r[2,0])
        ea = self.voltage * np.array([[self.front_left_power], [self.front_right_power], [self.back_left_power], [self.back_right_power]])
        eb = self.R @ Rotation.T @ self.q_rdot * self.motor_constant
    
        return (ea - eb) / self.armature_resistance
    
    def time_integrate(self, time_step):

        super().time_integrate(self.torque(), dt=time_step)
    
    @property
    def position(self):
        return self.q_r
    
    @property
    def velocity(self):
        return self.q_rdot
    
    @property
    def acceleration(self):
        return self.q_rddot
    
if __name__=="__main__":
    
    # simple test case of the Drivetrain class
    
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
    
    for i in range(1,5000):
        robot.set_powers(1, 1, 1, 1)
        robot.time_integrate(time_step)
        
        robot_position[i] = robot.position
        robot_velocity[i] = robot.velocity
        
    for i in range(5000, 7500):
        robot.set_powers(-1, -1, -1, -1)
        robot.time_integrate(time_step)
        
        robot_position[i] = robot.position
        robot_velocity[i] = robot.velocity
        
    for i in range(7500, N):
        robot.set_powers(1, -1, -1, 1)
        robot.time_integrate(time_step)
        
        robot_position[i] = robot.position
        robot_velocity[i] = robot.velocity
    
    fig1, axis = plt.subplots(2)
    fig2=plt.figure()
    
    robot.plot_evolution(t, robot_position, robot_velocity, fig=fig1)
    robot.plot_trajectory(robot_position, fig=fig2)