#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:54:42 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

class Outtake:
    HIGH_SPOOL_GEAR_RATIO = 3
    HIGH_SPOOL_RADIUS = 0.1
    LOW_SPOOL_RADIUS = 0.05

    FOURBAR_COM_RADIUS = .15
    FOURBAR_MASS = 1
    SLIDES_MASS = 1

    GRAVITY = 10

    
    def __init__(self, initial_angle = 0, initial_height = 0, initial_angular_velocity = 0, initial_linear_velocity = 0):
        self.angle = initial_angle
        self.height = initial_height
        self.angular_velocity = initial_angular_velocity
        self.linear_velocity = initial_linear_velocity
        

    def fourbar_torque(self, tension_left, tension_right):
        return self.HIGH_SPOOL_GEAR_RATIO * (tension_left - tension_right) * self.HIGH_SPOOL_RADIUS

    def slides_force(self, tension_left, tension_right):
        return tension_left + tension_right

    def linear_acceleration(self, tension_left, tension_right):
        return (self.FOURBAR_MASS * self.GRAVITY * np.sin(self.angle)**2
                - self.FOURBAR_MASS * self.FOURBAR_COM_RADIUS * self.angular_velocity**2 * np.cos(self.angle)
                - (self.SLIDES_MASS + self.FOURBAR_MASS) * self.GRAVITY
                + 1 / self.FOURBAR_COM_RADIUS * self.fourbar_torque(tension_left, tension_right) * np.sin(self.angle)
                + self.slides_force(tension_left, tension_right)) \
    / (self.SLIDES_MASS + self.FOURBAR_MASS * np.cos(self.angle)**2)

    def angular_acceleration(self, tension_left, tension_right):
        return (-self.GRAVITY / self.FOURBAR_COM_RADIUS * np.sin(self.angle)
                - 1 / self.FOURBAR_COM_RADIUS * self.linear_acceleration(tension_left, tension_right) * np.sin(self.angle)
                + self.fourbar_torque(tension_left, tension_right) / (self.FOURBAR_MASS * self.FOURBAR_COM_RADIUS**2))

    def time_integrate(self, torque_left, torque_right, time_step):
        tension_left = torque_left / self.LOW_SPOOL_RADIUS
        tension_right = torque_right / self.LOW_SPOOL_RADIUS

        linear_acceleration = self.linear_acceleration(tension_left, tension_right)
        angular_acceleration = self.angular_acceleration(tension_left, tension_right)

        self.angle += self.angular_velocity * time_step
        self.height += self.linear_velocity * time_step

        self.linear_velocity += linear_acceleration * time_step
        self.angular_velocity += angular_acceleration * time_step
    
    def get_state(self):
        return np.array([[self.angle],[self.height],[self.angular_velocity],[self.linear_velocity]])

    def to_string(self):
        return f'Height {self.height}, angle {self.angle.to("degrees")},' \
              f'Linear velocity {self.linear_velocity.to("metre/second")}, angular velocity {self.angular_velocity.to("degrees/second")}'

outtake = Outtake() 
torque_left = 0.55
torque_right = 0.55

N = 10000
dt = 0.005
simulation_time = np.arange(0,N)*dt
state = np.zeros((N,4,1))
for i in range(0,2500):
    state[i] = outtake.get_state()
    outtake.time_integrate(torque_left, torque_right, dt)
    # print(outtake.to_string())
for i in range(2500,N):
    state[i] = outtake.get_state()
    outtake.time_integrate(0.45, 0.45, dt)

plt.figure()
angles = state[:,0,0]
heights = state[:,1,0]
angular_velocity = state[:,2,0]
linear_velocity = state[:,3,0]
plt.plot(simulation_time, angles, 'r', label="angle")
plt.plot(simulation_time, heights, 'b', label="height")
plt.xlabel('time')
plt.ylabel('state')
plt.title("Outtake")
plt.legend()
plt.show()