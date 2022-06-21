#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:59:39 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

n = 1000
m = 10000

#Works for very large n*m but requires for loop
#X = np.zeros(m)
#for i in range(m):
#    xi = np.random.randint(1,7,n)
#    X[i] = np.mean(xi)

#Does not require for loop but uses lots of memory
xi = np.random.randint(1,7,size=(n,m))
X = np.mean(xi,axis=0)

plt.hist(X,30, density = True)

mu = 3.5
sigma = 1.71/np.sqrt(n)

xplot = np.arange(mu - 5*sigma, mu+ 5*sigma, 1E-3)
yplot = (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5/sigma**2 * np.square(xplot-mu))


plt.plot(xplot,yplot,'r')
plt.show()