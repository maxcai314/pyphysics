#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 11:15:00 2023

@author: maxcai
"""
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from motor_driven_drivetrain import Drivetrain
import pandas as pd
from scipy import interpolate

df = pd.read_csv('driving_around_log4.csv')
df['x_position'] = df['x_position'].div(39.37)  # convert from inch to meter
df['y_position'] = df['y_position'].div(39.37)
df['x_velocity'] = df['x_velocity'].div(39.37)
df['y_velocity'] = df['y_velocity'].div(39.37)

df_resampled = pd.DataFrame()
T = .1
df_resampled['time'] = np.arange(0, df['time'].max(), step=T)

for c in ['fl', 'fr', 'bl', 'br']:
    df_resampled[c] = interpolate.interp1d(df['time'], df[c], kind="previous", fill_value="extrapolate")(
        df_resampled['time'])
    df_resampled[c][0] = df_resampled[c][1]

for c in ['battery_voltage', 'x_position', 'y_position', 'angle', 'x_velocity', 'y_velocity', 'angular_velocity']:
    df_resampled[c] = interpolate.interp1d(df['time'], df[c], kind='cubic', fill_value="extrapolate")(
        df_resampled['time'])

df = df_resampled

df['x_acceleration'] = df['x_velocity'].diff().fillna(0) / T
df['y_acceleration'] = df['y_velocity'].diff().fillna(0) / T
df['angular_acceleration'] = df['angular_velocity'].diff().fillna(0) / T

startPos = np.array([0, 0, 0])  # x, y, angle
startVel = np.array([0, 0, 0])  # xVel, yVel, angleVel


def simulate(args, graph_velocity=False, graph_position=False):
    [I_z, I_w, motor_constant, ffl, ffr, bfl, bfr, fx, fy, fpsi] = args
    friction = np.array([[ffl], [ffr], [bfl], [bfr]])
    directional_friction = np.array([[fx], [fy], [fpsi]])
    robot = Drivetrain(motor_constant=motor_constant, I_z=I_z, I_w=I_w, friction=friction,
                       directional_friction=directional_friction, startPos=startPos.reshape((-1, 1)),
                       startVel=startVel.reshape((-1, 1)))

    robot_position = np.zeros((len(df), 3))
    robot_velocity = np.zeros((len(df), 3))
    robot_acceleration = np.zeros((len(df), 3))
    robot_position[0] = startPos
    robot_velocity[0] = startVel

    for i in range(len(df)):
        row = df.iloc[i]
        robot.voltage = row['battery_voltage']
        robot.set_powers(row['fl'], row['fr'], row['bl'], row['br'])
        robot.time_integrate(T)

        robot_position[i] = robot.position.reshape(-1)
        robot_velocity[i] = (robot.rotationmatrix(robot.position[2, 0]).T @ robot.velocity).reshape(-1)
        robot_acceleration[i] = (robot.rotationmatrix(robot.position[2, 0]).T @ robot.acceleration).reshape(-1)

    robot_position[:, 2] = ((robot_position[:, 2] + np.pi) % (2 * np.pi)) - np.pi

    if graph_velocity:
        plt.figure()
        plt.plot(df['time'], df['angular_velocity'], label='real angular vel')
        plt.plot(df['time'], robot_velocity[:, 2], label='predicted angular vel')

        plt.plot(df['time'], df['y_velocity'], label='real y vel')
        plt.plot(df['time'], robot_velocity[:, 1], label='predicted y vel')

        plt.plot(df['time'], df['x_velocity'], label='real x vel')
        plt.plot(df['time'], robot_velocity[:, 0], label='predicted x vel')

        plt.title("velocity")
        plt.legend()
        plt.show()

    if graph_position:
        plt.figure()
        plt.plot(df['time'], df['angle'], label='real angle')
        plt.plot(df['time'], robot_position[:, 2], label='predicted angle')
        #
        plt.plot(df['time'], df['x_position'], label='real x pos')
        plt.plot(df['time'], robot_position[:, 0], label='predicted x pos')

        plt.plot(df['time'], df['y_position'], label='real y pos')
        plt.plot(df['time'], robot_position[:, 1], label='predicted y pos')

        plt.title("position")
        plt.legend()
        plt.show()
    return np.sum(
        np.square((robot_velocity - np.array([df['x_velocity'], df['y_velocity'], df['angular_velocity']]).T)))


STEP = .0001


def grad(args, pool):  # use a thread pool to speed this up
    args_list = np.zeros((len(args), 2, len(args)))
    for i in range(len(args)):
        args_list[i, 0] = args
        args_list[i, 0, i] -= STEP
        args_list[i, 1] = args
        args_list[i, 1, i] += STEP

    results = np.array(pool.map(simulate, args_list.reshape((-1, len(args))))).reshape((len(args), 2))
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

    args = np.array(
        [1.99342709, 0.06464435, 0.26225076, 0.65752611, 0.6100913, 0.63821728, 1.58357671, 1.19970793, 1.00081366,
         1.20220801])
    simulate(args, graph_velocity=True, graph_position=True)

    with Pool(len(args) * 2 if DO_MULTITHREADING else 1) as p:
        for i in range(500):
            g = grad(args, p)

            print(g, repr(args), simulate(args))
            args -= g * .0001

    simulate(args, graph_velocity=True, graph_position=True)
