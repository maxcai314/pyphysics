#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:22:58 2022

Based on Master Thesis by I.M. Caireta 2019,
Model Predictive Control for a Mecanum-wheeled robot in Dynamical Environments
http://www.iri.upc.edu/files/academic/master_thesis/155-Master-thesis-document.pdf

@author: maxcai
"""
import numpy as np
import matplotlib.pyplot as plt

class Robot():
    def __init__(self,L=0.2,l=0.15,r=0.05,m=5.,I_z=3.,
                 I_w=[0.05,0.05,0.05,0.05],
                 S_geometric=None,d_geometric=None,alpha=None,
                 x_center=0., y_center=0.,
                 friction=0.1, q_r=None, q_rdot=None, d1=0.1,d2=-0.1,S3=-0.1,r_odo=0.05):
        self.L=L
        self.l=l
        self.r=r
        self.m=m
        self.I_z=I_z
        self.I_w = I_w
        self.S_geometric = S_geometric
        self.d_geometric = d_geometric
        self.alpha = alpha
        self.friction=friction
        self.q_r = q_r
        self.q_rdot = q_rdot
        self.d1 = d1
        self.d2 = d2
        self.S3 = S3
        self.r_odo = r_odo
        
        self.M_r = np.diag([self.m,self.m,self.I_z])
        self.M_w = np.diag(self.I_w)
        
        if self.S_geometric is None:
            self.S_geometric = np.array([self.L, self.L, -self.L, -self.L])
        if self.d_geometric is None:
            self.d_geometric = np.array([self.l, -self.l, self.l, -self.l])
        if self.alpha is None:
            self.alpha = [0.25 * np.pi, -0.25 * np.pi, -0.25 * np.pi, 0.25 * np.pi]
        
        if self.q_r is None:
            self.q_r = np.array([[0],[0],[0]])
        if self.q_rdot is None:
            self.q_rdot = np.array([[0],[0],[0]])
        
        self.q_rddot = np.array([[0],[0],[0]])
        
        self.x_center = x_center
        self.y_center = y_center
        
        self.S = self.S_geometric - self.x_center
        self.d = self.d_geometric - self.y_center
        
        self.q_d = np.zeros((3,1))
        self.R_d = 1./self.r_odo * np.array([[1,0,-self.d1],[1,0,-self.d2],[0,1,self.S3]])
        
        self.R = np.zeros((4,3))
        for i in range(4):
            self.R[i] = 1/self.r * np.array([1, -np.tan(self.alpha[i])**-1, -self.d[i]-self.S[i]*(np.tan(self.alpha[i])**-1)])
    
    def rotationmatrix(self,psi):
        return np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
    
    def rotationmatrixsmall(self,psi):
        return np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])

    def rotationmatrixdot(self,psi,psidot):
        return np.array([[-np.sin(psi),-np.cos(psi),0],[np.cos(psi),-np.sin(psi),0],[0,0,0]]) * psidot
    
    def H_analytic(self):
        return np.diag([self.m+self.I_w[0] * 4 * self.r**-2, self.m+self.I_w[0] * 4 * self.r**-2,self.I_z+ self.I_w[0] * 4 * (self.l+self.L)**2 * self.r**-2])
    
    def K_div_psidot_analytic(self):
        return np.matrix([[0,self.I_w[0] * 4 * self.r**-2 ,0],[-self.I_w[0] * 4 * self.r**-2,0,0],[0,0,0]])
    
    def get_aceleration(self, Gamma, angle, q_rdot):
        Rotation = self.rotationmatrix(angle)
        Rotationdot = self.rotationmatrixdot(angle,q_rdot[2,0])
        
        q_wdot = self.R @ Rotation.T @ q_rdot
        
        self.H = self.M_r + Rotation @ self.R.T @ self.M_w @ self.R @ Rotation.T
        self.K = Rotation @ self.R.T @ self.M_w @ self.R @ Rotationdot.T
        self.F_a = Rotation @ self.R.T @ (Gamma - q_wdot * self.friction)
        q_rddot = np.linalg.inv(self.H) @ (self.F_a - self.K @ q_rdot)
        return q_rddot
    
    def time_integrate(self, Gamma, dt=1E-2):
        self.q_rddot = self.get_aceleration(Gamma, self.q_r[2,0], self.q_rdot) #forward euler method
        prev_q_rdot = self.q_rdot
        self.q_rdot = self.q_rdot + self.q_rddot * dt
        self.q_r = self.q_r + prev_q_rdot * dt
        
    def time_integrate_steps(self, Gamma_list, N, dt=1E-2):
        self.N = N
        self.dt = dt
        self.t_list = np.arange(0, self.N*self.dt, self.dt)
        self.q_r_list = np.zeros((self.N,3,1))
        self.q_rdot_list = np.zeros((self.N,3,1))
        self.q_r_list[0] = self.q_r
        self.q_rdot_list[0] = self.q_rdot
        for i in range(1,N):
            self.time_integrate(Gamma_list[i-1], dt=dt)
            self.q_r_list[i] = self.q_r
            self.q_rdot_list[i] = self.q_rdot
    
    def get_wheel_vel(self, angle, q_rdot):
        return self.R @ (self.rotationmatrix(angle).T @ q_rdot)
    
    def get_wheel_vel_current(self):
        return self.get_wheel_vel(self.q_r[2,0], self.q_rdot)
    
    def get_odometry_vel(self, angle, q_rdot):
        return self.R_d @ (self.rotationmatrix(angle).T @ q_rdot)
    
    def simulate_odometry(self, q_r, q_rdot, dt=1E-2):
        self.q_d = self.q_d + self.get_odometry_vel(q_r[2,0], q_rdot) * dt
    
    def simulate_odometry_from_list(self):
        self.q_d_list = np.zeros((self.N,3,1))
        for i in range(1, self.N):
            self.q_d_list[i] = self.q_d_list[i-1] + self.get_odometry_vel(self.q_r_list[i-1,2,0], self.q_rdot_list[i-1]) * self.dt
    
    def predict_position_from_odometry(self, t_ext, q_d_funct,x0 = np.zeros((3,1))):
        N = t_ext.shape[0]
        q_r_predict = np.zeros((N,3,1))
        q_r_predict[0] = x0
        for i in range(1,N):
            dq_d = q_d_funct(t_ext[i]) - q_d_funct(t_ext[i-1])
            dq_r = self.rotationmatrix(q_r_predict[i-1,2,0]) @ (np.linalg.inv(self.R_d) @ dq_d) 
            q_r_predict[i] = q_r_predict[i-1] + dq_r
        return q_r_predict
    
    # def predict_position_from_odometry_fancy(self, t_ext, q_d_funct,x0 = np.zeros((3,1))):
    #     N = t_ext.shape[0]
    #     q_r_predict = np.zeros((N,3,1))
    #     q_r_predict[0] = x0
    #     for i in range(1,N):
    #         dt = t_ext[i]-t_ext[i-1]
    #         print("time:")
    #         print(t_ext[i])
    #         print("dt:")
    #         print(dt)
    #         q_rdot = (np.linalg.inv(self.R_d) @ q_d_funct(t_ext[i])) - (np.linalg.inv(self.R_d) @ q_d_funct(t_ext[i-1]))
    #         psidot = q_rdot[2]
    #         print('q_rdot is')
    #         print(q_rdot)
    #         posdot = np.array([[q_rdot[0,0]],[q_rdot[1,0]]])
    #         print('posdot is')
    #         print(posdot)
    #         integration_matrix = np.zeros((2,2))
    #         if psidot == 0:
    #             integration_matrix = np.identity(2) * dt
    #         else:
    #             integration_matrix = np.array([[np.sin(psidot),np.cos(psidot)-1],[1-np.cos(psidot),np.sin(psidot)]])[:,:,0]*dt/psidot
    #         print("integration matrix is")
    #         print(integration_matrix)
    #         print(integration_matrix.shape)
    #         dpos = self.rotationmatrixsmall(q_r_predict[i-1,2,0]) @ integration_matrix @ posdot
    #         dq_r = np.array([[dpos[0,0]],[dpos[1,0]],[psidot*dt]])
    #         q_r_predict[i] = q_r_predict[i-1] + dq_r
    #     return q_r_predict
    
    def control_heading_to_torque(self, q_rddot_heading):
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
   
    def plot_evolution(self, t, q_r, q_rdot, fig=None, block=False, show=True, colors=['b','r','g'], labels=['X','Y','Psi'], legends=True):
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2)
        elif len(fig.axes) < 2:
            if len(fig.axes) > 0:
                fig.axes[0].remove()
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
        else:
            ax1 = fig.axes[0]
            ax2 = fig.axes[1]
        ax1.title.set_text('Mecanum Wheeled Robot')
        ax2.set_xlabel('t')
        
        ax1.plot([0, np.max(t)],[0,0],'k')
        ax2.plot([0, np.max(t)],[0,0],'k')
        
        ax1.plot(t, q_r[:,0,0], colors[0], label=labels[0]+' pos')
        ax1.plot(t, q_r[:,1,0],colors[1], label=labels[1]+' pos')
        ax1.plot(t, q_r[:,2,0],colors[2], label=labels[2]+' pos')
        
        ax2.plot(t, q_rdot[:,0,0], colors[0], label=labels[0]+' vel')
        ax2.plot(t, q_rdot[:,1,0], colors[1], label=labels[1]+' vel')
        ax2.plot(t, q_rdot[:,2,0], colors[2], label=labels[2]+' vel')
        if legends:
            ax1.legend()
            ax2.legend()
        if show:
            plt.show(block=block)
    
    def plot_trajectory(self, q_r, fig=None, block=False, drawrobot=True, show=True, linecolor='b', label = None):
        if fig is None:
            fig = plt.figure()
        else:
            plt.figure(fig.number)
        
        xPos = q_r[:,0,0]
        yPos = q_r[:,1,0]

        maxd = np.max(np.abs(np.array([xPos,yPos])))
        
        plt.xlim(-1.2*maxd,1.2 * maxd)
        plt.ylim(-1.2*maxd,1.2 * maxd)
        if label is None:
            plt.plot(xPos,yPos,linecolor)
        else:
            plt.plot(xPos,yPos,linecolor,label=label)
            plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Robot Trajectory')
        if drawrobot:
            plt.plot(self.get_robot_outline(q_r[0])[:,0], self.get_robot_outline(q_r[0])[:,1],"purple")
            plt.plot(self.get_robot_outline(q_r[-1])[:,0], self.get_robot_outline(q_r[-1])[:,1],"red")
        plt.gca().set_aspect('equal')
        if show:
            plt.show(block=block)
    
    def get_state(self):
        return np.array([[self.q_r],[self.q_rdot]])
    
    def to_string(self):
        return f'Position {self.q_r}, Velocity {self.q_rdot}'

if __name__ == "__main__":
    q_r0 = np.array([[0],[0],[0]])
    q_rdot0 = np.array([[1],[1],[2]])
    
    robot = Robot(q_r=q_r0, q_rdot=q_rdot0)
    
    N = int(1E3)
    dt = 1E-2
    t = np.arange(0, N*dt, dt)
    
    Gamma = np.zeros((4,1))
    Gamma[0,0] = 2
    Gamma[1,0] = 1
    Gamma[2,0] = 2
    Gamma[3,0] = 1
    
    q_r = np.zeros((N,3,1))
    q_rdot = np.zeros((N,3,1))
    q_r[0] = q_r0
    q_rdot[0] = q_rdot0
    
    for i in range(1,N):
        robot.time_integrate(Gamma, dt=dt)
        q_r[i] = robot.q_r
        q_rdot[i] = robot.q_rdot
    
    fig1, axis = plt.subplots(2)
    fig2=plt.figure()
    
    robot.plot_evolution(t, q_r, q_rdot, fig=fig1)
    robot.plot_trajectory(q_r, fig=fig2)