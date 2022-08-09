#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:22:58 2022

@author: maxcai
"""
import numpy as np
import matplotlib.pyplot as plt

class Robot():
    def __init__(self,L=0.2,l=0.15,r=0.05,m=5,I_z=3,
                 I_w=[0.05,0.05,0.05,0.05],
                 S=None,d=None,alpha=None,
                 friction=10):
        self.L=L
        self.l=l
        self.r=r
        self.m=m
        self.I_z=I_z
        self.I_w = I_w
        self.S = S
        self.d = d
        self.alpha = alpha
        self.friction=friction
        
        if self.S==None:
            self.S = [self.L, self.L, -self.L, -self.L]
        if self.d==None:
            self.d = [self.l, -self.l, self.l, -self.l]
        if self.alpha==None:
            self.alpha = [0.25 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.25 * np.pi]
    
    def rotationmatrix(psi):
        return np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])

    def rotationmatrixdot(psi,psidot):
        return np.array([[-np.sin(psi),-np.cos(psi),0],[np.cos(psi),-np.sin(psi),0],[0,0,0]]) * psidot
    
    def H_analytic():
        #print(I_w[0] * 4 * r**-2 * q_rdot[2,0,-1])
        #print(m+I_w[0] * 4 * r**-2)
        #print(I_z+ I_w[0] * 4 * (l+L)**2 * r**-2)
        pass
    
    def K_div_psidot_analytic():
        pass
    
    def time_integrate(self,N,dt):
        self.N = N
        self.dt = dt
        pass
    
    def plot_evolution():
        pass
    
    def plot_trajectory():
        pass
    

robot = Robot()

N = int(1E3)
dt = 1E-2
t = np.arange(0, N*dt, dt)

#Robot Parameters
L, l = 0.2, 0.15
r = 0.05
m = 5
I_z = 3
I_w = np.array([0.05,0.05,0.05,0.05])
S = np.array([L, L, -L, -L])
d = np.array([l, -l, l, -l])
friction = 10#0
alpha = np.array([[0.25 * np.pi],[-0.25 * np.pi],[-0.25 * np.pi],[0.25 * np.pi]])

R = np.zeros((4,3))
for i in range(4):
    R[i] = 1/r * np.array([1, -np.tan(alpha[i])**-1, -d[i]-S[i]*(np.tan(alpha[i])**-1)])

M_r = np.diag([m,m,I_z])
M_w = np.diag(I_w)

q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[1],[0],[2]])
q_r = np.zeros((3,1,N))
q_rdot = np.zeros((3,1,N))
q_r[:,:,0] = q_r0
q_rdot[:,:,0] = q_rdot0

Gamma = np.zeros((4,1,N))


def rotationmatrix(psi):
    return np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])

def rotationmatrixdot(psi,psidot):
    return np.array([[-np.sin(psi),-np.cos(psi),0],[np.cos(psi),-np.sin(psi),0],[0,0,0]]) * psidot

for i in range(1,N):
    Rotation = rotationmatrix(q_r[2,0,i-1])
    Rotationdot = rotationmatrixdot(q_r[2,0,i-1],q_rdot[2,0,i-1])
    
    H = M_r + Rotation @ R.T @ M_w @ R @ Rotation.T
    K = Rotation @ R.T @ M_w @ R @ Rotationdot.T
    F_a = Rotation @ R.T @ Gamma[:,:,i-1]
    
    q_rddot = np.linalg.inv(H) @ (F_a - K @ q_rdot[:,:,i-1] - friction * q_rdot[:,:,i-1])
    q_rdot[:,:,i] = q_rdot[:,:,i-1] + q_rddot * dt
    q_r[:,:,i] = q_r[:,:,i-1] + q_rdot[:,:,i-1] * dt

print(I_w[0] * 4 * r**-2 * q_rdot[2,0,-1])
print(m+I_w[0] * 4 * r**-2)
print(I_z+ I_w[0] * 4 * (l+L)**2 * r**-2)

xPos = q_r[0,0,:]
yPos = q_r[1,0,:]

maxd = np.max(np.abs(np.array([xPos,yPos])))

print("\nFinal Position Matches:")
print(q_r[:,0,-1] - np.array([-2.14695516,  4.46602514,  2.55898749]) < 1E-8)

plt.figure(1)
plt.plot([0, np.max(t)],[0,0],'k')
plt.plot(t, q_r[0,0,:],'b', label='X position')
plt.plot(t, q_r[1,0,:],'r', label='Y position')
plt.plot(t, q_r[2,0,:],'g', label='Psi position')
plt.legend()
plt.xlabel('t')
plt.title('Mechanum Wheeled Robot')
plt.show()

plt.figure(2)
plt.plot([0, np.max(t)],[0,0],'k')
plt.plot(t, q_rdot[0,0,:],'b', label='X velocity')
plt.plot(t, q_rdot[1,0,:],'r', label='Y velocity')
plt.plot(t, q_rdot[2,0,:],'g', label='Psi velocity')
plt.legend()
plt.xlabel('t')
plt.title('Robot Velocity')
plt.show()

plt.figure(3)
#plt.axes().set_aspect('equal')
axis = plt.gca()
axis.set_aspect('equal')
plt.xlim(-1.1*maxd,1.1 * maxd)
plt.ylim(-1.1*maxd,1.1 * maxd)
plt.plot(xPos,yPos,'b')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Trajectory')