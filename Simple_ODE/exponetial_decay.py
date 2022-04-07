#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:21:14 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

lam = 0.5
y0 = 10
dt=0.1
t = np.arange(0, 10, dt)
y = y0 * np.exp(-lam * t)

#plotting
plt.plot(t, y)
plt.xlabel("t")
plt.ylabel("y")
plt.show()
