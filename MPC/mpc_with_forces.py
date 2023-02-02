#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
import numpy as np
setattr(np, 'float',  float)

from casadi import vertcat, horzcat
from forcespro.nlp.symbolicModel import SymbolicModel
from forcespro.nlp.solver import Solver

import forcespro
from forcespro import CodeOptions

from mpc_drive_simulator import DriveModel, get_configurable_parameters

time_lookahead = 1
frequency = 10
N = int(time_lookahead * frequency)

nin = 4
nstate = 6
npar = 1 + 2 * 10
if __name__ == '__main__':
    model = SymbolicModel(N)
    model.nvar = nin + nstate
    model.neq = nstate
    model.npar = npar

    robot = DriveModel()
    model.objective = robot.eval_obj
    model.eq = lambda z, p: forcespro.nlp.integrate(robot.continuous_dynamics, z[nin:], z[:nin], p,
                                                    integrator=forcespro.nlp.integrators.RK4,
                                                    stepsize=time_lookahead / frequency)

    model.E = np.hstack((np.zeros((nstate, nin)), np.eye(nstate)))

    model.lb = np.concatenate((np.ones(nin) * -1, np.ones(nstate) * -np.inf))
    model.ub = np.concatenate((np.ones(nin) * 1, np.ones(nstate) * np.inf))
    model.xinitidx = np.arange(nin, nin + nstate)

    codeoptions = CodeOptions("mecanum_mpc")

    codeoptions.solvemethod = 'PDIP_NLP'
    codeoptions.avx = -1
    codeoptions.sse = -1
    codeoptions.nlp.compact_code = 1
#    codeoptions.server = 'https://forces-test.embotech.com'
    codeoptions.cleanup = 0


    codeoptions.printlevel = 0
    codeoptions.optlevel = 2

    codeoptions.overwrite = 1
    m = model.generate_solver(codeoptions)
# ((uint64_t)*(int16_t*)(*(int64_t*)(&_returned_set_of_fingerprints.6495 + (((int64_t)var_10) << 3)) + (rdx_3 + rdx_3)))
