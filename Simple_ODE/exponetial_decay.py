#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:21:14 2022

@author: maxcai
"""

#ODE dy/dt = -lambda y
#Analytic solution, Euler forward, Euler backward, Crank-Nicolson Method

import numpy as np
import matplotlib.pyplot as plt

lam = 0.5
y0 = 10
dt = 0.01
t = np.arange(0, 20, dt)
y_anl = y0 * np.exp(-lam * t)

y_euler_fwd = np.zeros_like(t)
y_euler_fwd[0] = y0
for i in range(t.shape[0]-1):
    y_euler_fwd[i+1] = y_euler_fwd[i] * (1-lam*dt)

y_euler_bwd = np.zeros_like(t)
y_euler_bwd[0] = y0
for i in range (t.shape[0]-1):
    y_euler_bwd[i+1] = y_euler_bwd[i] / (1+lam*dt)

y_crank_nicholson = np.zeros_like(t)
y_crank_nicholson[0] = y0
for i in range(t.shape[0]-1):
    y_crank_nicholson[i+1] = y_crank_nicholson[i] * (1 - lam*dt/2) / (1 + lam*dt/2)

#plotting figure 1
plt.figure(1)
plt.plot(t, y_anl, 'r', label = 'Analytic')
plt.plot(t, y_euler_fwd, 'b', label = 'Euler Forward')
plt.plot(t, y_euler_bwd, 'g', label = 'Euler Backward')
plt.plot(t, y_crank_nicholson, 'm', label = 'Crank-Nicolson')
plt.legend()
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution')
plt.show()

plt.figure(2)
plt.plot(t, np.abs(y_euler_fwd - y_anl), 'b', label = 'Euler Forward')
plt.plot(t, np.abs(y_euler_bwd - y_anl), 'g', label = 'Euler Backward')
plt.plot(t, np.abs(y_crank_nicholson - y_anl), 'm', label = 'Crank-Nicolson')
plt.legend()
plt.xlabel('t')
plt.ylabel('dy')
plt.title('Error')
plt.yscale('log')
plt.show()