#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 10:03:26 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain
import pandas as pd
from scipy import interpolate


class DriveLogReader():
    def __init__(self, directory_name, motor_constant=0.36827043, I_z=1.55687398, I_w=np.ones(4)*0.0549806, friction=np.array([[0.05758405], [0.00923156], [0.06716439], [0.02307628]])):
        self.log = pd.read_csv(directory_name).to_dict('records')
        
        self.x_pos_real = np.array([d['x_position'] for d in self.log]) / 39.37 # convert from inch to meter
        self.y_pos_real = np.array([d['y_position'] for d in self.log]) / 39.37
        self.angle_real = np.array([d['angle'] for d in self.log])
        self.x_vel_real = np.array([d['x_velocity'] for d in self.log]) / 39.37 # velocities are all relative
        self.y_vel_real = np.array([d['y_velocity'] for d in self.log]) / 39.37
        self.angle_vel_real = np.array([d['angular_velocity'] for d in self.log])
        
        self.fl = np.array([d['fl'] for d in self.log]) # logged motor powers
        self.fr = np.array([d['fr'] for d in self.log])
        self.bl = np.array([d['bl'] for d in self.log])
        self.br = np.array([d['br'] for d in self.log])
        
        self.voltage_real = [d['battery_voltage'] for d in self.log]
        
        self.times_real = np.array([d['time'] for d in self.log])
        
        self.max_time = np.round(np.max(self.times_real))

        self.voltage_function = interpolate.interp1d(self.times_real, self.voltage_real, kind="cubic", fill_value="extrapolate")

        self.flPower = interpolate.interp1d(self.times_real, self.fl, kind="cubic", fill_value="extrapolate")
        self.frPower = interpolate.interp1d(self.times_real, self.fr, kind="cubic", fill_value="extrapolate")
        self.blPower = interpolate.interp1d(self.times_real, self.bl, kind="cubic", fill_value="extrapolate")
        self.brPower = interpolate.interp1d(self.times_real, self.br, kind="cubic", fill_value="extrapolate")

        self.x_vel = interpolate.interp1d(self.times_real, self.x_vel_real, kind="cubic", fill_value="extrapolate")
        self.y_vel = interpolate.interp1d(self.times_real, self.y_vel_real, kind="cubic", fill_value="extrapolate")
        self.angle_vel = interpolate.interp1d(self.times_real, self.angle_vel_real, kind="cubic", fill_value="extrapolate")
        
        self.pos_real = np.zeros((self.times_real.shape[0],3,1))
        self.pos_real[:,0,0] = self.x_pos_real
        self.pos_real[:,1,0] = self.y_pos_real
        self.pos_real[:,2,0] = self.angle_real
        
        self.vel_real = np.zeros((self.times_real.shape[0],3,1))
        self.vel_real[:,0,0] = self.x_vel_real
        self.vel_real[:,1,0] = self.y_vel_real
        self.vel_real[:,2,0] = self.angle_vel_real
        
        startPos = self.pos_real[0] # x, y, angle
        startVel = self.vel_real[0] # xVel, yVel, angleVel
        
        self.robot = Drivetrain(motor_constant=motor_constant, I_z=I_z, I_w=I_w, friction=friction, startPos=startPos, startVel=startVel)

    def simulate_from_logs(self, time_step = 1E-3, derivative_step = 0.1, angle_wrap = True, relative_vel = True):
        N = int(self.max_time/time_step)
        self.t = np.arange(0, N*time_step, time_step)
        
        self.pos_sim = np.zeros((N,3,1))
        self.vel_sim = np.zeros((N,3,1))
        self.accel_sim = np.zeros((N,3,1))
        
        #simulation from motor powers
        for i in range(N):
            currentTime = self.t[i]
            self.robot.voltage = self.voltage_function(currentTime)
            self.robot.set_powers(self.flPower(currentTime),self.frPower(currentTime),self.blPower(currentTime),self.brPower(currentTime))
            self.robot.time_integrate(time_step)
            
            rotation = np.identity(3)
            if relative_vel:
                rotation = self.robot.rotationmatrix(self.robot.position[2,0]).T
            
            self.pos_sim[i] = self.robot.position
            self.vel_sim[i] = rotation @ self.robot.velocity
            self.accel_sim[i] = rotation @ self.robot.acceleration
        
        if angle_wrap:
            self.pos_sim[:,2,0] = ((self.pos_sim[:,2,0]+np.pi)%(2*np.pi)) - np.pi # angle wrap
        
        # calculate acceleration of physical robot
        self.x_accel_real = np.zeros((N,1))
        self.y_accel_real = np.zeros((N,1))
        self.angle_accel_real = np.zeros((N,1))
        for i in range(N):
            currentTime = self.t[i]
            self.x_accel_real[i] = (self.x_vel(currentTime + derivative_step) - self.x_vel(currentTime))/derivative_step
            self.y_accel_real[i] = (self.y_vel(currentTime + derivative_step) - self.y_vel(currentTime))/derivative_step
            self.angle_accel_real[i] = (self.angle_vel(currentTime + derivative_step) - self.angle_vel(currentTime))/derivative_step
    
    def plot_evolution(self, legends = False):
        fig1, axis = plt.subplots(2)
        self.robot.plot_evolution(self.t, self.pos_sim, self.vel_sim, fig=fig1, show=False, legends=legends, colors=['b','r','g'], labels=['sim X', 'sim Y','sim Psi'])
        self.robot.plot_evolution(self.times_real, self.pos_real, self.vel_real, fig=fig1, show=True, legends=legends, colors=['c','y','m'], labels=['real X', 'real Y','real Psi'])
    
    def plot_trajectory(self):
        fig2 = plt.figure()
        self.robot.plot_trajectory(self.pos_real, linecolor='c', fig=fig2, show=False)
        self.robot.plot_trajectory(self.pos_sim, linecolor='b', fig=fig2, show=True)
    
    def plot_acceleration(self, showX = True, showY = True, showAngle = True):
        plt.figure()
        if showX:
            plt.plot(self.t, self.x_accel_real,'c', label='X accel real')
            plt.plot(self.t, self.accel_sim[:,0,0],'b', label='X accel simulate')
        if showY:
            plt.plot(self.t, self.y_accel_real,'y', label='Y accel real')
            plt.plot(self.t, self.accel_sim[:,1,0],'r', label='Y accel simulate')
        if showAngle:
            plt.plot(self.t, self.angle_accel_real,'m', label='Psi accel real')
            plt.plot(self.t, self.accel_sim[:,2,0],'g', label='Psi accel simulate')
        
        plt.legend()
        plt.title('Acceleration')
        plt.xlabel('t')
        plt.show()

if __name__ == '__main__':
    # simple example code
    simulation = DriveLogReader('drive_samples/driving_around_log_slower_8.csv')
    simulation.simulate_from_logs(time_step=1E-3, angle_wrap=False,relative_vel=True)
    simulation.plot_evolution(legends = True)
    simulation.plot_trajectory()
    simulation.plot_acceleration()
    
    
    
    
    
    
    