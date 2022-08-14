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
                 friction=0.1):
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
        
        self.M_r = np.diag([self.m,self.m,self.I_z])
        self.M_w = np.diag(self.I_w)
        
        if self.S==None:
            self.S = [self.L, self.L, -self.L, -self.L]
        if self.d==None:
            self.d = [self.l, -self.l, self.l, -self.l]
        if self.alpha==None:
            self.alpha = [0.25 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.25 * np.pi]
        
        self.R = np.zeros((4,3))
        for i in range(4):
            self.R[i] = 1/self.r * np.array([1, -np.tan(self.alpha[i])**-1, -self.d[i]-self.S[i]*(np.tan(self.alpha[i])**-1)])
    
    def rotationmatrix(self,psi):
        return np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])

    def rotationmatrixdot(self,psi,psidot):
        return np.array([[-np.sin(psi),-np.cos(psi),0],[np.cos(psi),-np.sin(psi),0],[0,0,0]]) * psidot
    
    def H_analytic(self):
        return np.diag([self.m+self.I_w[0] * 4 * self.r**-2, self.m+self.I_w[0] * 4 * self.r**-2,self.I_z+ self.I_w[0] * 4 * (self.l+self.L)**2 * self.r**-2])
    
    def K_div_psidot_analytic(self):
        return np.matrix([[0,self.I_w[0] * 4 * self.r**-2 ,0],[self.I_w[0] * 4 * self.r**-2,0,0],[0,0,0]])
    
    def time_integrate(self,q_r0, q_rdot0, Gamma, N, dt=1E-2):
        self.N = N
        self.dt = dt
        self.t = np.arange(0, self.N*self.dt, self.dt)
        self.q_r = np.zeros((N,3,1))
        self.q_rdot = np.zeros((N,3,1))
        self.q_r[0] = q_r0
        self.q_rdot[0] = q_rdot0
        
        for i in range(1,N):
            Rotation = self.rotationmatrix(self.q_r[i-1,2,0])
            Rotationdot = self.rotationmatrixdot(self.q_r[i-1,2,0],self.q_rdot[i-1,2,0])
            
            q_wdot = self.R @ Rotation.T @ self.q_rdot[i-1]
            
            self.H = self.M_r + Rotation @ self.R.T @ self.M_w @ self.R @ Rotation.T
            self.K = Rotation @ self.R.T @ self.M_w @ self.R @ Rotationdot.T
            self.F_a = Rotation @ self.R.T @ (Gamma[i-1] - q_wdot * self.friction)
            
            self.q_rddot = np.linalg.inv(self.H) @ (self.F_a - self.K @ self.q_rdot[i-1])
            self.q_rdot[i] = self.q_rdot[i-1] + self.q_rddot * self.dt
            self.q_r[i] = self.q_r[i-1] + self.q_rdot[i-1] * self.dt
        
    def convert_target_control_heading_to_torque(self, q_rddot_heading):
        return self.R @ q_rddot_heading
   
    def get_robot_outline(self, state):
        rotation = np.array([[np.cos(state[2]),-np.sin(state[2])],[np.sin(state[2]),np.cos(state[2])]])[:,:,0]
        pos = np.array([[state[0,0]],[state[1,0]]])
        output = np.zeros((5,2,1))
        output[0] = pos + rotation @ np.array([[self.L],[self.l]])
        output[1] = pos + rotation @ np.array([[self.L],[-self.l]])
        output[2] = pos + rotation @ np.array([[-self.L],[-self.l]])
        output[3] = pos + rotation @ np.array([[-self.L],[self.l]])
        output[4] = pos + rotation @ np.array([[self.L],[self.l]])
        return output
   
    def plot_evolution(self, fig=None, ax1=None, ax2=None, block=False):
        if fig==None:
            fig, (ax1, ax2) = plt.subplots(2)
        
        ax1.plot([0, np.max(self.t)],[0,0],'k')
        ax1.plot(self.t, self.q_r[:,0,0],'b', label='X position')
        ax1.plot(self.t, self.q_r[:,1,0],'r', label='Y position')
        ax1.plot(self.t, self.q_r[:,2,0],'g', label='Psi position')
        ax1.legend()
        ax1.title.set_text('Mechanum Wheeled Robot')
        
        ax2.plot([0, np.max(self.t)],[0,0],'k')
        ax2.plot(self.t, self.q_rdot[:,0,0],'b', label='X velocity')
        ax2.plot(self.t, self.q_rdot[:,1,0],'r', label='Y velocity')
        ax2.plot(self.t, self.q_rdot[:,2,0],'g', label='Psi velocity')
        ax2.legend()
        ax2.set_xlabel('t')
        
        plt.show(block=block)
    
    def plot_trajectory(self, fig=None, block=False, drawrobot = True):
        if fig==None:
            fig = plt.figure()
        
        xPos = self.q_r[:,0,0]
        yPos = self.q_r[:,1,0]

        maxd = np.max(np.abs(np.array([xPos,yPos])))
        
        axis = plt.gca()
        axis.set_aspect('equal')
        plt.xlim(-1.1*maxd,1.1 * maxd)
        plt.ylim(-1.1*maxd,1.1 * maxd)
        plt.plot(xPos,yPos,'b')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Trajectory')
        if drawrobot:
            plt.plot(self.get_robot_outline(self.q_r[0])[:,0],self.get_robot_outline(self.q_r[0])[:,1],"purple")
            plt.plot(self.get_robot_outline(self.q_r[-1])[:,0],self.get_robot_outline(self.q_r[-1])[:,1],"red")
        plt.show(block=block)

if __name__ == "__main__":
    robot = Robot()
    
    N = int(1E3)
    Gamma = np.zeros((N,4,1))
    Gamma[:,0,0] = 2
    Gamma[:,1,0] = 1
    Gamma[:,2,0] = 2
    Gamma[:,3,0] = 1
    q_r0 = np.array([[0],[0],[0]])
    q_rdot0 = np.array([[1],[1],[2]])
    
    robot.time_integrate(q_r0, q_rdot0, Gamma, N)
    robot.plot_evolution()
    robot.plot_trajectory()