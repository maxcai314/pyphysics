#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 22:59:41 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

N = 20000
mu, sigma = 0, 0.2
X = np.random.normal(mu, sigma, N)

xplot = np.arange(-1, 1, 1E-3)
yplot = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5/sigma**2 * np.square(xplot-mu))

plt.hist(X,50, density=True)
plt.plot(xplot,yplot,'r')
plt.show()