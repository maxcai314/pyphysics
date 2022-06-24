#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:59:39 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

n = 2000
mu = 0
sigma = 1

X = np.random.normal(mu,sigma,(2,n))
Sigma_X = np.cov(X)
print("Sigma_X")
print(Sigma_X)

A = np.array([[2,0],[1,1]])
Y = np.matmul(A,X)
Sigma_Y = np.cov(Y)
print("Sigma_Y")
print(Sigma_Y)

B = np.array([[2,0],[0,np.sqrt(2)]])
Z = np.matmul(B,X)
Sigma_Z = np.cov(Z)
print("Sigma_Z")
print(Sigma_Z)

plt.figure(1)
plt.scatter(X[0],X[1],color='blue',s=20,alpha=0.3)
plt.scatter(Y[0],Y[1],color='red',s=20,alpha=0.3)
#plt.scatter(Z[0],Z[1],color='green',s=20,alpha=0.3)
plt.title("2D Gaussian")
plt.axis('equal')

plt.figure(2)
plt.hist(Y[1],30,density = True)
plt.title("Y2")

plt.figure(3)
plt.hist(Z[1],30,density = True)
plt.title("Z2")

plt.show()
