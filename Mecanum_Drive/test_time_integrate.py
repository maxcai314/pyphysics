#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 19:52:06 2022

@author: maxcai
"""

import numpy as np
from mecanum_drive import Robot

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

ref_q_r = np.array([[ 0.73906968],[ 0.83730122],[-5.36311989]])
max_error = np.max(np.abs(robot.q_r[-1] - ref_q_r))
if max_error<1E-8:
    print("test passed")
else:
    print("test failed")
