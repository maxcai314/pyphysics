#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
import numpy as np
from forcespro.nlp.symbolicModel import SymbolicModel
from forcespro.nlp.solver import Solver
from forcespro import CodeOptions

from PyPhysics.Mecanum_Drive.mpc_drive_simulator import DriveModel, get_configurable_parameters

time_lookahead = 1
frequency = 10

nin = 4
nstate = 6
npar = 1 + 2 * 10

model = SymbolicModel(time_lookahead * frequency)
model.nvar = nin + nstate
model.neq = nstate
model.npar = npar

robot = DriveModel()
model.LSobjective = robot.eval_obj
model.continuous_dynamics = robot.continuous_dynamics
model.E = np.hstack((np.zeros((nstate, nin)), np.eye(nstate)))

model.lb = np.concatenate((np.ones(nin) * -1, np.ones(nstate) * -np.inf))
model.ub = np.concatenate((np.ones(nin) * 1, np.ones(nstate) * np.inf))
model.xinitidx = np.arange(nin, nin + nstate)

codeoptions = CodeOptions()

codeoptions.nlp.integrator.type = 'ERK4'
codeoptions.nlp.integrator.Ts = 1 / frequency
codeoptions.nlp.integrator.nodes = 10

codeoptions.printlevel = 2

codeoptions.overwrite = 1
#m = model.generate_solver(codeoptions)
m = Solver.from_directory("FORCES_NLP_solver")
battery_voltage = 12

NUM_ITERATIONS = 100
configurable_parameters = get_configurable_parameters()
all_configurable_params = np.zeros((NUM_ITERATIONS, len(1 + configurable_parameters)))
all_configurable_params[:, 1:] = configurable_parameters
all_configurable_params[:, 0] = battery_voltage

initial_position = np.array([0, 0, 0])
initial_velocity = np.array([0, 0, 0])
problem = {
    "x0":
}