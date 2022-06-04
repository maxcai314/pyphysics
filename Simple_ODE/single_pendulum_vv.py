#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:09:37 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

#dd_theta + g/l sin(theta) = 0

g = 10
l = 10
m = 1
dt = 1E-3
theta0 = 1
omega0 = 0
t = np.arange(0, 40, dt)
theta = np.zeros_like(t)
omega = np.zeros_like(t)
energy = np.zeros_like(t)
theta[0] = theta0
omega[0] = omega0

#Velocity Verlet
ddtheta = -g/l * np.sin(theta[0]) #second derivative of theta
for i in range(t.shape[0]-1):
    theta[i+1] = theta[i] + dt * omega[i] + 0.5 * dt**2 *ddtheta
    omega_halfstep = omega[i] + 0.5 * dt * ddtheta #temporary variable
    ddtheta = -g/l * np.sin(theta[i+1]) #next timestep
    omega[i+1] = omega_halfstep + 0.5 *dt * ddtheta

energy = 0.5 * m * np.square(omega) * l**2 - (m*g*l*np.cos(theta))
print("Energy change = ", energy[-1] - energy[0])
#local energy conservation is on the order of dt^2
#global energy conservation is linear with dt
#When dt = 1E-4, dE = 0.24 (at time 100)
#When dt = 1E-3, dE = 3.37
#When dt = 1E-2, dE = 41.94

plt.figure(1)
plt.plot(t, theta, 'b', label = 'theta')
plt.plot(t, omega, 'g', label = 'omega')
plt.plot([0, np.max(t)],[0,0],'k')
plt.legend()
plt.xlabel('t')
plt.title('Pendulum')
plt.show()

plt.figure(2)
plt.plot(t, energy)
plt.xlabel('t')
plt.ylabel('energy')
plt.title('Pendulum')
plt.show()
