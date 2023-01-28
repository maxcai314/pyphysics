#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain
import pandas as pd
from scipy import interpolate

T = .1


@dataclass
class DataSeries:
    name: str
    time: np.ndarray
    battery_voltage: np.ndarray
    x_position: np.ndarray
    y_position: np.ndarray
    angle: np.ndarray
    x_velocity: np.ndarray
    y_velocity: np.ndarray
    angular_velocity: np.ndarray
    x_acceleration: np.ndarray
    y_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    fl: np.ndarray
    fr: np.ndarray
    bl: np.ndarray
    br: np.ndarray

    def __post_init__(self):
        self.time = np.array(self.time)
        self.battery_voltage = np.array(self.battery_voltage)
        self.x_position = np.array(self.x_position)
        self.y_position = np.array(self.y_position)
        self.angle = np.array(self.angle)
        self.x_velocity = np.array(self.x_velocity)
        self.y_velocity = np.array(self.y_velocity)
        self.angular_velocity = np.array(self.angular_velocity)
        self.x_acceleration = np.array(self.x_acceleration)
        self.y_acceleration = np.array(self.y_acceleration)
        self.angular_acceleration = np.array(self.angular_acceleration)
        self.fl = np.array(self.fl)
        self.fr = np.array(self.fr)
        self.bl = np.array(self.bl)
        self.br = np.array(self.br)

    @staticmethod
    def _load_df(filename):
        df = pd.read_csv(filename)
        df['x_position'] = df['x_position'].div(39.37)
        df['y_position'] = df['y_position'].div(39.37)
        df['x_velocity'] = df['x_velocity'].div(39.37)
        df['y_velocity'] = df['y_velocity'].div(39.37)
        df_resampled = pd.DataFrame()
        df_resampled['time'] = np.arange(0, df['time'].max(), step=T)
        for c in ['fl', 'fr', 'bl', 'br']:
            df_resampled[c] = interpolate.interp1d(df['time'], df[c], kind="previous", fill_value="extrapolate")(
                df_resampled['time'])
            df_resampled[c][np.isnan(df_resampled[c])] = 0

        for c in ['battery_voltage', 'x_position', 'y_position', 'angle', 'x_velocity', 'y_velocity',
                  'angular_velocity']:
            df_resampled[c] = interpolate.interp1d(df['time'], df[c], kind='cubic', fill_value="extrapolate")(
                df_resampled['time'])

        df_resampled['x_acceleration'] = np.gradient(df_resampled['x_velocity'], T)
        df_resampled['y_acceleration'] = np.gradient(df_resampled['y_velocity'], T)
        df_resampled['angular_acceleration'] = np.gradient(df_resampled['angular_velocity'], T)

        return df_resampled

    @staticmethod
    def from_csv(filename):
        df = DataSeries._load_df(filename)
        return DataSeries(filename, df['time'], df['battery_voltage'], df['x_position'], df['y_position'], df['angle'],
                          df['x_velocity'], df['y_velocity'], df['angular_velocity'], df['x_acceleration'],
                          df['y_acceleration'], df['angular_acceleration'], df['fl'], df['fr'], df['bl'], df['br'])

    def plot(self, ax):
        ax.plot(self.x_position, self.y_position, label='Path')
        ax.plot(self.x_position[0], self.y_position[0], 'ro', label='Start')
        ax.plot(self.x_position[-1], self.y_position[-1], 'go', label='End')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.legend()
        ax.grid()
        ax.set_aspect('equal')

    def plot_velocity(self, ax):
        ax.plot(self.time, self.x_velocity, label='X Velocity')
        ax.plot(self.time, self.y_velocity, label='Y Velocity')
        ax.plot(self.time, self.angular_velocity, label='Angular Velocity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend()
        ax.grid()

    def plot_acceleration(self, ax):
        ax.plot(self.time, self.x_acceleration, label='X Acceleration')
        ax.plot(self.time, self.y_acceleration, label='Y Acceleration')
        ax.plot(self.time, self.angular_acceleration, label='Angular Acceleration')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m/s^2)')
        ax.legend()
        ax.grid()

    def plot_voltage(self, ax):
        ax.plot(self.time, self.battery_voltage, label='Battery Voltage')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        ax.grid()

    def plot_motor_speeds(self, ax):
        ax.plot(self.time, self.fl, label='FL')
        ax.plot(self.time, self.fr, label='FR')
        ax.plot(self.time, self.bl, label='BL')
        ax.plot(self.time, self.br, label='BR')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Motor Speed (rad/s)')
        ax.legend()
        ax.grid()

    def __len__(self):
        return len(self.time)

    def __getitem__(self, item):
        return np.array([self.time[item], self.battery_voltage[item], self.x_position[item], self.y_position[item],
                         self.angle[item], self.x_velocity[item], self.y_velocity[item], self.angular_velocity[item],
                         self.x_acceleration[item], self.y_acceleration[item], self.angular_acceleration[item],
                         self.fl[item], self.fr[item], self.bl[item], self.br[item]])


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
        robot.time_integrate(T)

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
