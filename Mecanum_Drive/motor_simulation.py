#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 10:35:09 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

class Motor:
    # INDUCTANCE = 0
    MOMENT_OF_INERTIA = 2
    ARMATURE_RESISTANCE = 1.6
    TORQUE_CONSTANT = 0.366
    EMF_CONSTANT = 0.366
    VISCOUS_FRICTION_COEFFICIENT = 0.04
    
    def __init__(self):
        self.torque = 0
        self.angular_vel = 0
    
    def torque_dot(self,voltage):
        return ((self.VISCOUS_FRICTION_COEFFICIENT * self.TORQUE_CONSTANT) * voltage + (self.VISCOUS_FRICTION_COEFFICIENT * self.ARMATURE_RESISTANCE - self.EMF_CONSTANT * self.TORQUE_CONSTANT) * self.torque)/(self.ARMATURE_RESISTANCE * self.MOMENT_OF_INERTIA)
    
    def angular_vel_dot(self, voltage):
        return ((self.EMF_CONSTANT * voltage) - (self.VISCOUS_FRICTION_COEFFICIENT * self.ARMATURE_RESISTANCE) * self.angular_vel)/(self.MOMENT_OF_INERTIA * self.ARMATURE_RESISTANCE)
    
    def time_integrate(self, voltage, time_step):
        torque_dot = self.torque_dot(voltage)
        angular_vel_dot = self.angular_vel_dot(voltage)
        
        self.torque += torque_dot * time_step
        self.angular_vel += angular_vel_dot * time_step
    
    def get_state(self):
        return np.array([[self.angular_vel],[self.torque]])
    
    def to_string(self):
        return f'Angular Velocity {self.angular_vel}, Torque {self.torque}'

if __name__ == "__main__":
    motor = Motor()
    
    N = 100000
    dt = 0.005
    simulation_time = np.arange(0,N)*dt
    state = np.zeros((N,2,1))
    for i in range(0,2500):
        state[i] = motor.get_state()
        motor.time_integrate(12, dt)
        # print(outtake.to_string())
    for i in range(2500,N):
        state[i] = motor.get_state()
        motor.time_integrate(0, dt)
    
    plt.figure()
    anglular_vel = state[:,0,0]
    torque = state[:,1,0]
    plt.plot(simulation_time, anglular_vel, 'r', label="angular velocity")
    plt.plot(simulation_time, torque, 'b', label="torque")
    plt.xlabel('time')
    plt.ylabel('state')
    plt.title("Motor Test")
    plt.legend()
    plt.show()