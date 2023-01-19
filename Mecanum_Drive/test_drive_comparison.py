#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""
import glob
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool

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
            df_resampled[c][0] = df_resampled[c][1]

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


def simulate(sample, args, graph_velocity=False, graph_position=False):
    if len(args) < 3 + 4 + 3:
        args = np.append(args, np.zeros(3 + 4 + 3 - len(args)))  # add zeros for missing parameters

    [I_z, I_w, motor_constant, ffl, ffr, bfl, bfr, fx, fy, fpsi] = args
    friction = np.array([[ffl], [ffr], [bfl], [bfr]])
    directional_friction = np.array([[fx], [fy], [fpsi]])

    startPos = np.array([sample.x_position[0], sample.y_position[0], sample.angle[0]])
    startVel = np.array([sample.x_velocity[0], sample.y_velocity[0], sample.angular_velocity[0]])

    robot = Drivetrain(motor_constant=motor_constant, I_z=I_z, I_w=I_w, friction=friction,
                       directional_friction=directional_friction, startPos=startPos.reshape((-1, 1)),
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

    robot_position[:, 2] = ((robot_position[:, 2] + np.pi) % (2 * np.pi)) - np.pi

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
    return np.sum(
        np.square((robot_velocity - np.array([sample.x_velocity, sample.y_velocity, sample.angular_velocity]).T)))


STEP = .0001


def grad(sample, args, pool):  # use a thread pool to speed this up
    args_list = np.zeros((len(args), 2, len(args)))
    for i in range(len(args)):
        args_list[i, 0] = args
        args_list[i, 0, i] -= STEP
        args_list[i, 1] = args
        args_list[i, 1, i] += STEP

    results = np.array(pool.map(partial(simulate, sample), args_list.reshape((-1, len(args))))).reshape((len(args), 2))
    derivatives = (results[:, 1] - results[:, 0]) / (2 * STEP)
    return derivatives / np.linalg.norm(derivatives)


def grad_simple(args):
    derivatives = np.zeros(len(args))
    for i in range(len(args)):
        args[i] -= STEP
        a = simulate(args)
        args[i] += 2 * STEP
        b = simulate(args)
        args[i] -= STEP
        derivatives[i] = (b - a) / (2 * STEP)

    return derivatives / np.linalg.norm(derivatives)


if __name__ == '__main__':
    DO_MULTITHREADING = True  # this might kill your computer
    samples = [DataSeries.from_csv(f) for f in glob.glob('drive_samples/*.csv')]
    args = np.array([1.99335099, 0.07465765, 0.26251987, 0, 0, 0, 0])  # 1.19970748, 1.0008127 , 1.20225137])

    with Pool(len(args) * 2 if DO_MULTITHREADING else 1) as p:
        for epoch_num in range(500):
            costs = np.zeros(len(samples))

            for j in range(len(samples)):
                g = grad(samples[j], args, p)
                costs[j] = simulate(samples[j], args)
                args -= g * .0001

            print(f"epoch {epoch_num}, average cost: {np.mean(costs)}, args: {args}")

    simulate(args, graph_velocity=True, graph_position=True)
