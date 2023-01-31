#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 15:02:49 2023

@author: maxcai
"""
import numpy as np
from casadi import vertcat, horzcat
from forcespro.nlp.symbolicModel import SymbolicModel
from forcespro.nlp.solver import Solver

import forcespro
from forcespro import CodeOptions

from PyPhysics.Mecanum_Drive.mpc_drive_simulator import DriveModel, get_configurable_parameters

time_lookahead = 1
frequency = 10
N = int(time_lookahead * frequency)

nin = 4
nstate = 6
npar = 1 + 2 * 10

model = SymbolicModel(N)
model.nvar = nin + nstate
model.neq = nstate
model.npar = npar

robot = DriveModel()
model.objective = robot.eval_obj
model.eq = lambda z, p: forcespro.nlp.integrate(robot.continuous_dynamics, z[nin:], z[:nin], p,
                                                integrator=forcespro.nlp.integrators.RK4,
                                                stepsize=1 / frequency)

model.E = np.hstack((np.zeros((nstate, nin)), np.eye(nstate)))

model.lb = np.concatenate((np.ones(nin) * -1, np.ones(nstate) * -np.inf))
model.ub = np.concatenate((np.ones(nin) * 1, np.ones(nstate) * np.inf))
model.xinitidx = np.arange(nin, nin + nstate)

codeoptions = CodeOptions("mecanum_mpc")

codeoptions.solvemethod = 'PDIP_NLP'

codeoptions.printlevel = 2

codeoptions.overwrite = 1
# m = model.generate_solver(codeoptions)
m = Solver.from_directory("mecanum_mpc")


def run_solver(
        target_position,
        battery_voltage=12,
        target_velocity=horzcat(0, 0, 0),

        motor_voltage_weights=horzcat(0, 0, 0, 0),
        position_weights=horzcat(1, 1, 0),
        velocity_weights=horzcat(0, 0, 0),
):
    global all_params
    target_position = np.array(target_position).reshape((3, 1))
    target_velocity = np.array(target_velocity).reshape((3, 1))
    motor_voltage_weights = np.array(motor_voltage_weights).reshape((4, 1))
    position_weights = np.array(position_weights).reshape((3, 1))
    velocity_weights = np.array(velocity_weights).reshape((3, 1))

    configurable_parameters = get_configurable_parameters(
        target_position=target_position,
        target_velocity=target_velocity,
        motor_voltage_weights=motor_voltage_weights,
        position_weights=position_weights,
        velocity_weights=velocity_weights,
    )
    all_params = np.zeros((N, 1 + len(configurable_parameters)))
    all_params[:, 0] = battery_voltage
    all_params[:, 1:] = configurable_parameters

    all_params[-1, 16:20:2] = 1  # on the last step, we want to be stopped

    initial_position = np.array([0, 0, 0])
    initial_velocity = np.array([0, 0, 0])

    initial_guess = np.ones((N, nin + nstate))
    initial_state = np.concatenate((initial_position, initial_velocity))
    problem = {
        "x0": initial_guess,
        "xinit": initial_state,
        "all_parameters": all_params.reshape((-1,)),
    }

    output_dict, exitflag, info = m.solve(problem)

    output = np.zeros((N, nin + nstate))
    for entry in output_dict:
        if entry.startswith("x"):
            output[int(entry[1:]) - 1] = output_dict[entry]

    return output, info


if __name__ == '__main__':
    output, info = run_solver(
        vertcat(1, 1, 0),
        battery_voltage=12,
        target_velocity=vertcat(0, 0, 0),

        motor_voltage_weights=vertcat(0, 0, 0, 0),
        position_weights=vertcat(1, 1, 0),
        velocity_weights=vertcat(0, 0, 0),
    )

    position = output[:, 4:7]
    velocity = output[:, 7:10]
    time = np.arange(N) / frequency

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time, position[:, 0], label="X")
    ax[0].plot(time, position[:, 1], label="Y")
    ax[0].legend()
    ax[0].set_ylabel("Position (m)")

    ax[1].plot(time, velocity[:, 0], label="X")
    ax[1].plot(time, velocity[:, 1], label="Y")
    ax[1].legend()
    ax[1].set_ylabel("Velocity (m/s)")

    fig_inputs, ax_inputs = plt.subplots()
    ax_inputs.plot(time, output[:, :4])
    ax_inputs.legend(["FL", "FR", "BL", "BR"])
    ax_inputs.set_ylabel("Motor Voltage (V)")
    ax_inputs.set_xlabel("Time (s)")


    plt.show()
