#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
import aerosandbox.numpy as np
import casadi
import numba

junction_spacing = .5969 # 23.5"
ground_junction_diameter = .1524 # 6"
pole_washer_diameter = .0762 # 3"
field_width = 3.6576 # 144"
robot_radius = .34 * np.sqrt(2) / 2

NOTHING = 0
GROUND_ = 1
POLE___ = 2
field_grid = [
	[NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING],
	[NOTHING, GROUND_, POLE___, GROUND_, POLE___, GROUND_, NOTHING],
	[NOTHING, POLE___, POLE___, POLE___, POLE___, POLE___, NOTHING],
	[NOTHING, GROUND_, POLE___, GROUND_, POLE___, GROUND_, NOTHING],
	[NOTHING, POLE___, POLE___, POLE___, POLE___, POLE___, NOTHING],
	[NOTHING, GROUND_, POLE___, GROUND_, POLE___, GROUND_, NOTHING],
	[NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING, NOTHING],
]

# convert the field grid to a list of (x, y) coordinates
ground_junctions = []
poles = []
for y in range(len(field_grid)):
	for x in range(len(field_grid[y])):
		if field_grid[y][x] == GROUND_:
			ground_junctions.append((x * junction_spacing, y * junction_spacing))
		elif field_grid[y][x] == POLE___:
			poles.append((x * junction_spacing, y * junction_spacing))

ground_junctions = np.array(ground_junctions)
poles = np.array(poles)

numba.njit = lambda f: f  # disable njit for forces
numba.njit.disabled = True

from drive_simulation import OptimisationParameters, RobotState

setattr(np, 'float', float)


def inequalities(z: RobotState):
	robot_x = z.x
	robot_y = z.y

	robot_position = np.array([robot_x, robot_y])
	ground_junction_distances = (np.sum((ground_junctions - np.stack([robot_position] * len(ground_junctions))) ** 2, axis=1)) # broadcasting is broken in casadi
	pole_distances = (np.sum((poles - np.stack([robot_position] * len(poles))) ** 2, axis=1))

	all_ineq_values = np.concatenate((ground_junction_distances, pole_distances))

	return all_ineq_values

def ineq_lb():
	ground_junction_lb = (robot_radius + ground_junction_diameter / 2) ** 2
	pole_lb = (robot_radius + pole_washer_diameter / 2) ** 2
	all_ineq_lb = np.concatenate((np.ones(len(ground_junctions)) * ground_junction_lb, np.ones(len(poles)) * pole_lb))
	return all_ineq_lb

def ineq_ub():
	all_ineq_ub = np.ones(len(ground_junctions) + len(poles)) * np.inf
	return all_ineq_ub

from forcespro.nlp.symbolicModel import SymbolicModel

import forcespro
from forcespro import CodeOptions

time_lookahead = 1
frequency = 10
N = int(time_lookahead * frequency)

nin = 4
nstate = 6
npar = OptimisationParameters.num_parameters()
if __name__ == '__main__':
	model = SymbolicModel(N)
	model.nvar = nin + nstate
	model.neq = nstate
	model.npar = npar

# 	model.nh = len(ground_junctions) + len(poles)
# 	model.ineq = lambda z, p: inequalities(RobotState.from_array(z))
# 	model.hl = ineq_lb()
# 	model.hu = ineq_ub()

	model.objective = lambda z, p: OptimisationParameters.from_array(p).objective(RobotState.from_array(z))
	integrator_stepsize = 0.1
	continuous_dynamics = lambda x, u, p: OptimisationParameters.from_array(p).model.continuous_dynamics(RobotState.from_array(casadi.vertcat(u, x)))
	model.eq = lambda z, p: forcespro.nlp.integrate(continuous_dynamics, z[nin:nin + nstate], z[0:nin], p, integrator=forcespro.nlp.integrators.RK4, stepsize=integrator_stepsize)
	model.E = np.hstack((np.zeros((nstate, nin)), np.eye(nstate)))

	model.lb = np.concatenate((np.ones(nin) * -1, np.ones(nstate) * -np.inf))
	model.ub = np.concatenate((np.ones(nin) * 1, np.ones(nstate) * np.inf))
	model.xinitidx = range(nin, nstate + nin)

	codeoptions = CodeOptions("mecanum_mpc")

	codeoptions.solvemethod = 'PDIP_NLP'
	codeoptions.server = 'https://forces-6-0-1.embotech.com'
	codeoptions.cleanup = 0

	codeoptions.printlevel = 2
	codeoptions.optlevel = 0

	codeoptions.overwrite = 1

	codeoptions.sse = -1
	codeoptions.avx = -1

	m = model.generate_solver(codeoptions)
