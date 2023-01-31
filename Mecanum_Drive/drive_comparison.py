#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""

import numpy as np
import matplotlib.pyplot as plt

from PyPhysics.Mecanum_Drive.mecanum_data import DataSeries
from motor_driven_drivetrain import Drivetrain


def simulate(sample, graph_velocity=False, graph_position=False):
    startPos = np.array([sample.x_position[0], sample.y_position[0], sample.angle[0]])
    startVel = np.array([sample.x_velocity[0], sample.y_velocity[0], sample.angular_velocity[0]])

    robot = Drivetrain(motor_constant=0.36827043, I_z=1.55687398, I_w=np.ones(4)*0.0549806, friction=np.array([0.05758405, 0.00923156, 0.06716439, 0.02307628]).reshape((-1, 1)),
                       startPos=startPos.reshape((-1, 1)),
                       startVel=startVel.reshape((-1, 1)))

    robot_position = np.zeros((len(sample), 3))
    robot_velocity = np.zeros((len(sample), 3))
    robot_acceleration = np.zeros((len(sample), 3))
    robot_position[0] = startPos
    robot_velocity[0] = startVel

    for i in range(len(sample)):
        robot.voltage = sample.battery_voltage[i]
        robot.set_powers(sample.fl[i], sample.fr[i], sample.bl[i], sample.br[i])
        robot.time_integrate(DataSeries.T)

        robot_position[i] = robot.position.reshape(-1)
        robot_velocity[i] = (robot.rotationmatrix(robot.position[2, 0]).T @ robot.velocity).reshape(-1)
        robot_acceleration[i] = (robot.rotationmatrix(robot.position[2, 0]).T @ robot.acceleration).reshape(-1)

    if graph_velocity:
        plt.figure()
        plt.plot(sample.time, sample.x_velocity, label='X Velocity (Measured)')
        plt.plot(sample.time, sample.y_velocity, label='Y Velocity (Measured)')

        plt.plot(sample.time, robot_velocity[:, 0], label='X Velocity (Simulated)')
        plt.plot(sample.time, robot_velocity[:, 1], label='Y Velocity (Simulated)')

        plt.title("velocity")
        plt.legend()
        plt.show()

    if graph_position:
        plt.figure()
        plt.plot(sample.time, sample.angle, label='Angle (Measured)')
        plt.plot(sample.time, sample.x_position, label='X Position (Measured)')
        plt.plot(sample.time, sample.y_position, label='Y Position (Measured)')

        plt.plot(sample.time, robot_position[:, 2], label='Angle (Simulated)')
        plt.plot(sample.time, robot_position[:, 0], label='X Position (Simulated)')
        plt.plot(sample.time, robot_position[:, 1], label='Y Position (Simulated)')

        plt.title("position")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    simulate(DataSeries.from_csv("drive_samples/driving_around_log_slower_8.csv"), graph_velocity=True, graph_position=False)
