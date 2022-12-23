import numpy as np
from sympy import *
from sympy.abc import t
from sympy.vector import gradient
from motor_simulation import Motor
print("a")


left_motor = Motor(.005, 1.6, 0.36, 0.37)
right_motor = Motor(.005, 1.6, 0.36, 0.37)

gravity = 9.81
non_rotating_mass, fourb_mass, fourb_com_distance = var('slides_mass, fourb_mass, fourb_com_distance')

fourb_height = Function('fourb_height')(t)
fourb_angle = Function('fourb_angle')(t)

kinetic_energy = .5 * (
        non_rotating_mass * fourb_height.diff() ** 2
        + fourb_mass * (
                fourb_com_distance * fourb_angle.diff() ** 2
                + fourb_height.diff() ** 2
                + 2 * fourb_height.diff() * fourb_com_distance * fourb_angle.diff() * sin(fourb_angle)
        )
)

potential_energy = non_rotating_mass * gravity * fourb_height \
                   + fourb_mass * gravity * (fourb_height - fourb_com_distance * cos(fourb_angle))

position = Matrix([fourb_height, fourb_angle])
lagrangian = (kinetic_energy - potential_energy).simplify()
velocity = position.diff()
momenta = lagrangian.diff(velocity).simplify()
hamiltonian = (velocity.dot(momenta) - lagrangian).simplify()
# scipy segfaults here

velocity = Matrix([hamiltonian.diff(t) / momenta[0].diff(t), hamiltonian.diff(t) / momenta[1].diff(t)])
momenta = Matrix([-hamiltonian.diff(t) / velocity[0].diff(t), -hamiltonian.diff(t) / velocity[1].diff(t)])