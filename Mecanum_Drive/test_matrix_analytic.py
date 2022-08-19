#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 20:58:15 2022

@author: maxcai
"""

import numpy as np
from mecanum_drive import Robot

robot = Robot()

N = int(10)
Gamma = np.zeros((N,4,1))

q_r0 = np.array([[0],[0],[0]])
q_rdot0 = np.array([[1],[1],[2]])

robot.time_integrate(q_r0, q_rdot0, Gamma, N)

print()
print("H = ")
print(robot.H_analytic())
max_error_H = np.max(np.abs(robot.H_analytic() - robot.H))
if max_error_H < 1E-12:
    print("test H passed")
else:
    print("test H failed")

print()
print("K / psidot = ")
print(robot.K_div_psidot_analytic())
max_error_K = np.max(np.abs(robot.K_div_psidot_analytic() - robot.K/robot.q_rdot[-2,2]))
if max_error_K < 1E-12:
    print("test H passed")
else:
    print("test H failed")
