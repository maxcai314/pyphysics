#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 22:18:25 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

R = 0.1 #Variance of noise
Q = 0.2 #Variance of measurement noise
x0 = 0
Sigma0 = 1
N = 100 #number of steps
A = 1
C = 1
x = np.zeros(N)
epsilon = np.random.normal(0,np.sqrt(R),N)
delta = np.random.normal(0,np.sqrt(Q),N)
mu = np.zeros(N)
Sigma = np.zeros(N)

t = np.arange(N)
z = np.zeros(N)

#simulation
x[0] = x0
for i in range(1,N):
    x[i] = A * x[i-1] + epsilon[i]

#measurement
for i in range(1,N):
    z[i] = C * x[i] + delta[i]

#kalman filter
mu[0] = x0
Sigma[0] = Sigma0
for i in range (1,N):
    #prediction
    mu_pred = A * mu[i-1]
    Sigma_pred = A * Sigma[i-1] * A + R
    Sigma[i] = 1./(C * 1./Q * C + 1./Sigma_pred)
    mu[i] = Sigma[i] * (C * 1./Q * z[i] + 1./Sigma_pred * mu_pred)
    
    

plt.figure(1)
plt.plot(t,x,'r',label="Real State")
plt.plot(t,mu,'b',label= "Estimated State")
plt.title("Kalman Filter")
plt.legend()
#plt.plot(t,mu-np.sqrt(Sigma),'k')
#plt.plot(t,mu+np.sqrt(Sigma),'k')
plt.show()