#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:59:39 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

n = 10000
m = 10000

X = np.zeros(m)
for i in range(m):
    xi = np.random.randint(1,7,n)
    X[i] = np.mean(xi)

plt.hist(X,30, density = True)

mu = 3.5
sigma = 1.71/np.sqrt(n)

xplot = np.arange(mu - 5*sigma, mu+ 5*sigma, 1E-3)
yplot = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5/sigma**2 * np.square(xplot-mu))


plt.plot(xplot,yplot,'r')
plt.show()