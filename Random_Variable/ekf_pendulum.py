#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 13:33:16 2022

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

class Pendulum():
    def __init__(self,g=10,l=1,dt=1E-2,decay = 0.5,x0 = np.array([[1],[0]]),Sigma0 = np.array([[0,0],[0,0]]),R = 1E-9 * np.identity(2),Q = 0.2):
        #todo: turn R and Q into covariance matrices instead of numbers
        self.g = g
        self.l = l
        self.k = self.g/self.l
        self.decay = decay
        self.dt = dt
        self.x0 = x0
        self.C = np.array([[1,0]]) #row vector
        #todo: eliminate N and make computing real-time
        #B = 0
        #todo: add control function
        self.x = np.zeros((2,1))
        self.mu = np.zeros((2,1))
        self.Sigma = np.zeros((2,2))
        self.z = 0
        self.x[0] = self.x0
        self.mu[0] = self.x0
        self.Sigma[0] = Sigma0
        self.control = 0
        self.R = R
        self.Q = Q
        self.record_x = np.array([])
        self.record_mu = np.array([])
        self.record_z = np.array([])
        #TODO: record values
        #todo: move to separate fucntion to generate real-time
        #make independant on N
        self.epsilon = np.zeros((2,1))
        self.delta = 0
    
    def evolve(self,x,control):
        output = np.zeros((2,1))
        output[0] = x[0] + self.dt * x[1] - self.dt**2 * self.k * np.sin(x[0])
        output[1] = x[1] - self.dt * self.k * np.sin(x[0]) + self.dt * control
        return output
    
    def simulate(self,control):
        self.x = self.evolve(self.x,self.control)
        self.epsilon = np.array([np.random.multivariate_normal(np.array([0,0]),self.R)]).T
        self.delta = np.random.normal(0,np.sqrt(self.Q))
        self.x += self.epsilon
        self.control = -self.decay * self.x

    def measure(self):
        self.z = self.C @ self.x + self.delta
    
    def kalman_filter(self):
        mu_pred = np.zeros((2,1))
        G = np.array([[1-self.k*self.dt**2 * np.cos(self.mu[0]),self.dt],[-self.k * self.dt * np.cos(self.mu[0]), 1-self.decay * self.dt]]) #Jacobian of evolution function
        #prediction
        mu_pred = self.evolve(self.mu,self.control)
        Sigma_pred = G @ self.Sigma @ G.T + self.R
        #kalman gain
        K = Sigma_pred @ self.C.T * (self.C @ Sigma_pred @ self.C.T + self.Q)**-1
        #measurement
        self.Sigma = (np.identity(2)- K @ self.C) @ Sigma_pred
        self.mu = mu_pred + K @ (self.z-self.C @ mu_pred)
    

np.random.seed(seed=10)

pendulum = Pendulum()
pendulum.simulate()
pendulum.measure()
pendulum.kalman_filter()

t = pendulum.dt * np.arange(pendulum.N)

plt.figure(1)

plt.plot(t, pendulum.z, 'r', label = 'theta measured')
plt.plot(t, pendulum.x[:,0], 'b', label = 'theta')
plt.plot(t, pendulum.mu[:,0], 'g', label = 'theta estimate')
# plt.plot(t, pendulum.x[:,1], 'b', label = 'omega')
# plt.plot(t, pendulum.mu[:,1], 'g', label = 'omega estimate')
plt.plot([0, np.max(t)],[0,0],'k')
plt.legend()
plt.xlabel('t')
plt.title('Pendulum EKF')

# plt.figure(2)
# plt.plot(t, pendulum.x[:,0]-pendulum.mu[:,0], 'r', label = 'theta error')
# plt.plot(t, pendulum.x[:,1]-pendulum.mu[:,1], 'g', label = 'omega error')
# plt.plot([0, np.max(t)],[0,0],'k')
# plt.legend()
# plt.xlabel('t')
# plt.title('Pendulum EKF error')

plt.figure(3)
plt.plot(t,pendulum.Sigma[:,0,0],'b',label = 'theta variance')
plt.plot(t,pendulum.Sigma[:,0,1],'g',label = 'theta and omega covariance')
plt.plot(t,pendulum.Sigma[:,1,1],'r',label = 'omega variance')
plt.legend()
plt.xlabel('t')
plt.title('EKF Covariance')

plt.show()

    