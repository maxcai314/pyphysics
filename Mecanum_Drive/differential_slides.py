#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:54:42 2022

@author: maxcai
"""

import pint
import numpy as np

u = pint.UnitRegistry(system='SI')
Nm = u.N * u.m

HIGH_SPOOL_GEAR_RATIO = 3
HIGH_SPOOL_RADIUS = 2 * u.inch
LOW_SPOOL_RADIUS = 2 * u.inch

FOURBAR_COM_RADIUS = .1 * u.metre
FOURBAR_MASS = .5 * u.kilogram
SLIDES_MASS = .5 * u.kilogram

GRAVITY = 10 * u.metre / u.second**2


class Outtake:
    def __init__(self):
        self.angle = 0 * u.radians
        self.angular_velocity = 0 * u.radian / u.second
        self.height = 0 * u.metre
        self.linear_velocity = 0 * u.metre / u.second

    def fourbar_torque(self, tension_left, tension_right):
        return HIGH_SPOOL_GEAR_RATIO * (tension_left - tension_right) * HIGH_SPOOL_RADIUS

    def slides_force(self, tension_left, tension_right):
        return tension_left + tension_right

    def linear_acceleration(self, tension_left, tension_right):
        return (FOURBAR_MASS * GRAVITY * np.sin(self.angle)**2
                - FOURBAR_MASS * FOURBAR_COM_RADIUS * self.angular_velocity**2 * np.cos(self.angle)
                - (SLIDES_MASS + FOURBAR_MASS) * GRAVITY
                + 1 / FOURBAR_COM_RADIUS * self.fourbar_torque(tension_left, tension_right) * np.sin(self.angle)
                + self.slides_force(tension_left, tension_right)) \
            / (SLIDES_MASS + FOURBAR_MASS * np.cos(self.angle)**2)

    def angular_acceleration(self, tension_left, tension_right):
        return (-GRAVITY / FOURBAR_COM_RADIUS * np.sin(self.angle)
                - 1 / FOURBAR_COM_RADIUS * self.linear_acceleration(tension_left, tension_right) * np.sin(self.angle)
                + self.fourbar_torque(tension_left, tension_right) / (FOURBAR_MASS * FOURBAR_COM_RADIUS**2)).to('rad / second**2')

    def simulate(self, torque_left, torque_right, time_step):
        tension_left = torque_left / LOW_SPOOL_RADIUS
        tension_right = torque_right / LOW_SPOOL_RADIUS

        linear_acceleration = self.linear_acceleration(tension_left, tension_right)
        angular_acceleration = self.angular_acceleration(tension_left, tension_right)

        self.angle += self.angular_velocity * time_step
        self.height += self.linear_velocity * time_step

        self.linear_velocity += linear_acceleration * time_step
        self.angular_velocity += angular_acceleration * time_step

    def to_string(self):
        return f'Height {self.height}, angle {self.angle.to("degrees")},' \
              f'Linear velocity {self.linear_velocity.to("metre/second")}, angular velocity {self.angular_velocity.to("degrees/second")}'

outtake = Outtake()
torque_left = 1 * Nm
torque_right = 1 * Nm

for i in np.arange(0, 10, step = .1):
    outtake.simulate(torque_left, torque_right, .1 * u.seconds)
    print(outtake.to_string())









