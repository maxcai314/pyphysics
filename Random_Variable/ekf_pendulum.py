#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 13:33:16 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

class Pendulum():
    def __init__(self,g=10,l=1,dt=1E-2,x0 = np.array([[3.1],[0]]),Sigma0 = np.array([[1,0],[0,1]]),R = 1E-9 * np.identity(2),Q = 0.2):
        #todo: turn R and Q into covariance matrices instead of numbers
        self.g = g
        self.l = l
        self.k = self.g/self.l
        self.dt = dt
        self.x0 = x0
        self.C = np.array([[1,0]]) #row vector
        #todo: eliminate N and make computing real-time
        #B = 0
        #u = np.zeros((N,2)) #Control
        #todo: add controls
        self.N = 1000
        self.x = np.zeros((self.N,2,1))
        self.mu = np.zeros((self.N,2,1))
        self.Sigma = np.zeros((self.N,2,2))
        self.z = np.zeros(self.N)
        self.x[0] = self.x0
        self.mu[0] = self.x0
        self.Sigma[0] = Sigma0
        self.R = R
        self.Q = Q
        #todo: move to separate fucntion to generate real-time
        #make independant on N
        self.epsilon = np.zeros((self.N,2,1))
        self.delta = np.zeros(self.N)
        
        for i in range(self.N):
            #np.random.multivariate_normal generates row vectors, so each has to be set and transposed one by one
            self.epsilon[i] = np.array([np.random.multivariate_normal(np.array([0,0]),R)]).T
        
        self.delta = np.random.normal(0,Q,self.N)
    
    def evolve(self,x):
        output = np.zeros((2,1))
        output[0] = x[0] + self.dt * x[1] - self.dt**2 * self.k * np.sin(x[0])
        output[1] = x[1] - self.dt * self.k * np.sin(x[0])
        return output
    
    def simulate(self):
        for i in range(1,self.N):
            self.x[i] = self.evolve(self.x[i-1])
            self.x[i] += self.epsilon[i]
    
    def measure(self):
        for i in range(0,self.N):
            self.z[i] = self.C @ self.x[i] + self.delta[i]
    
    def kalman_filter(self):
        mu_pred = np.zeros((2,1))
        for i in range (1,self.N):
            G = np.array([[1-self.k*self.dt**2 * np.cos(self.mu[i-1,0]),self.dt],[-self.k * self.dt * np.cos(self.mu[i-1,0]), 1]]) #Jacobian of evolution function
            #prediction
            mu_pred = self.evolve(self.mu[i-1])
            Sigma_pred = G @ self.Sigma[i-1] @ G.T + self.R
            #kalman gain
            K = Sigma_pred @ self.C.T * (self.C @ Sigma_pred @ self.C.T + self.Q)**-1
            #measurement
            self.Sigma[i] = (np.identity(2)- K @ self.C) @ Sigma_pred
            self.mu[i] = mu_pred + K @ (self.z[i]-self.C @ mu_pred)
    

# R = 1E-9 #Variance of noise, no convariance matrix because both noise values currently satisfy same distribution
# Q = 0.2 #Variance of measurement noise
# x0 = np.array([[3.1],[0]]) #theta, omega
# Sigma0 = np.array([[1,0],[0,1]])
# N = 1000 #number of steps
# g = 10
# l = 1
# k = g/l
# dt = 1E-2

# #B = 0
# C = np.array([[1,0]]) #row vector

# x = np.zeros((N,2,1))
# #u = np.zeros((N,2)) #Control
np.random.seed(seed=10)
# epsilon = np.random.normal(0,np.sqrt(R),(N,2,1))
# delta = np.random.normal(0,np.sqrt(Q),N)
# mu = np.zeros((N,2,1))
# Sigma = np.zeros((N,2,2))

#todo: add control
#for i in range(N):
#    u[i] = 0.1


# z = np.zeros(N)

# #simulation
# x[0] = x0
# for i in range(1,N):
#     x[i] = evolve(x[i-1])
#     x[i] += epsilon[i]

# #measurement
# for i in range(0,N):
#     z[i] = np.matmul(C, x[i]) + delta[i]
#     #only theta is measured
    
#kalman filter (kalman gain)
# mu[0] = x0
# Sigma[0] = Sigma0
# mu_pred = np.zeros((2,1))
# for i in range (1,N):
#     G = np.array([[1-k*dt**2 * np.cos(mu[i-1,0]),dt],[-k * dt * np.cos(mu[i-1,0]), 1]]) #Jacobian of evolution function
#     #prediction
#     mu_pred = evolve(mu[i-1])
#     Sigma_pred = G @ Sigma[i-1] @ G.T + R
#     #kalman gain
#     K = Sigma_pred @ C.T * (C @ Sigma_pred @ C.T + Q)**-1
#     #measurement
#     Sigma[i] = (np.identity(2)- K @ C) @ Sigma_pred
#     mu[i] = mu_pred + K @ (z[i]-C @ mu_pred)

#print(np.max(np.abs(mu[:,0]-mu_k[:,0])))
#print((Sigma[-1]-Sigma_k[-1]))

pendulum = Pendulum()
pendulum.simulate()
pendulum.measure()
pendulum.kalman_filter()

t = pendulum.dt * np.arange(pendulum.N)

plt.figure(1)

plt.plot(t, pendulum.z, 'r', label = 'theta measured')
plt.plot(t, pendulum.x[:,0], 'b', label = 'theta')
plt.plot(t, pendulum.mu[:,0], 'g', label = 'theta estimate')
#plt.plot(t, x[:,1], 'g', label = 'omega')
plt.plot([0, np.max(t)],[0,0],'k')
plt.legend()
plt.xlabel('t')
plt.title('Pendulum EKF')

plt.figure(2)
plt.plot(t,pendulum.Sigma[:,0,0],'b',label = 'theta variance')
plt.plot(t,pendulum.Sigma[:,1,0],'m')
plt.plot(t,pendulum.Sigma[:,0,1],'g',label = 'theta and omega covariance')
plt.plot(t,pendulum.Sigma[:,1,1],'r',label = 'omega variance')
plt.legend()
plt.xlabel('t')
plt.title('EKF Covariance')

plt.show()

    