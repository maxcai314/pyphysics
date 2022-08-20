#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 22:39:17 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#dd_theta + g/l sin(theta) = 0

g = 10.
l = 10.
dt = 0.5
theta0 = 3.1
omega0 = 0
y0 = [theta0,omega0]
t = np.arange(0.,100.,dt)

def pend(y,t,g,l):
    theta, omega = y
    dydt = [omega,  - g/l * np.sin(theta)]
    return dydt

y = odeint(pend, y0, t, args=(g,l))

plt.figure(1)
plt.plot(t, y[:,0], 'b', label = 'theta')
plt.plot(t, y[:,1], 'g', label = 'omega')
plt.plot([0, np.max(t)],[0,0],'k')
plt.legend()
plt.xlabel('t')
plt.title('Pendulum')
plt.show()
