#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 16:09:37 2022

@author: maxcai
"""

import numpy as np
import sys
from casadi import *

import do_mpc

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.stages.Model(model_type)

theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))
omega = model.set_variable(var_type='_x', var_name='omega', shape=(1,1))
u     = model.set_variable(var_type='_u', var_name='torque', shape=(1,1))

#dd_theta + g/l sin(theta) = u

np.random.seed(10)
g = 10
l = 10
m = 1
theta0 = 1
omega0 = 0
dt = 1E-1

N = 100

use_own_plot = True

model.set_rhs('theta',omega)
model.set_rhs('omega', -g/l * sin(theta) + u)

model.set_expression(expr_name='cost',expr=(theta**2 + omega**2))

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon' : 10,
    'n_robust' : 0,
    'open_loop' : 0,
    't_step' : dt,
    'state_discretization' : 'collocation',
    'collocation_deg' : 3,
    'collocation_ni' : 1,
    'store_full_solution' : True,
    'nlpsol_opts' : {'ipopt.linear_solver' : 'mumps'}
}

mpc.set_param(**setup_mpc)

mterm = model.aux['cost']
lterm = model.aux['cost']

mpc.set_objective(mterm=mterm,lterm=lterm)

mpc.set_rterm(torque=1e-4)

mpc.bounds['upper','_x','theta'] = np.pi

mpc.bounds['lower','_x','theta'] = -np.pi

mpc.bounds['upper','_u','torque'] = 0.2

mpc.bounds['lower','_u','torque'] = -0.2

mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)

simulator = do_mpc.simulator.Simulator(model)

simulator.set_param(t_step = dt)
simulator.setup()

x0 = np.array([[theta0],[omega0]])
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

u0 = np.array([[0.]])
x_traj = np.zeros((N,2,1))
x_traj[0] = x0

for k in range(N):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    x_traj[k] = x0



if use_own_plot:
    import matplotlib.pyplot as plt
    
    t = np.arange(0,N)
    
    plt.figure(1)
    plt.plot(t, x_traj[:,0], 'b', label = 'theta')
    plt.plot(t, x_traj[:,1], 'g', label = 'omega')
    plt.plot([0, np.max(t)],[0,0],'k')
    plt.legend()
    plt.xlabel('t')
    plt.title('Pendulum')
    plt.show(block=False)
else:
    from matplotlib import rcParams
    rcParams['axes.grid'] = True
    rcParams['font.size'] = 18
    
    import matplotlib.pyplot as plt
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data,figsize=(16,9))
    graphics.plot_results()
    graphics.reset_axes()
    plt.show(block=False)