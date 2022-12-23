import numpy as np
from matplotlib import pyplot as plt

from motor_simulation import Motor
from differential_slides import DifferentialSlides, GRAVITY


class Outtake(DifferentialSlides):
    def __init__(self, initial_angle=0, initial_height=0, initial_angular_velocity=0, initial_linear_velocity=0):
        super().__init__(initial_angle, initial_height, initial_angular_velocity, initial_linear_velocity)
        self.left = Motor(.005, 1.6, 0.36, 0.37)
        self.right = Motor(.005, 1.6, 0.36, 0.37)
        self.battery_voltage = 12

    def time_integrate(self, left_motor_power, right_motor_power, time_step):
        left_motor_voltage = self.battery_voltage * left_motor_power
        right_motor_voltage = self.battery_voltage * right_motor_power

        self.left.time_integrate(left_motor_voltage, time_step)
        self.right.time_integrate(right_motor_voltage, time_step)

        super().time_integrate(self.left.torque, self.right.torque, time_step)

    @property
    def current(self):
        return self.left.current + self.right.current


if __name__ == "__main__":
    outtake = Outtake(initial_angle=np.pi / 2, initial_height=0)
    timestep = 0.01
    total_time = 5

    steps = int(total_time / timestep)

    time = np.zeros(steps)
    height = np.zeros(steps)
    angle = np.zeros(steps)
    current = np.zeros(steps)
    force = np.zeros(steps)
    torque = np.zeros(steps)
    angular_velocity = np.zeros(steps)

    current_time = 0
    for index in range(0, steps):
        time[index] = current_time
        height[index] = outtake.height
        angle[index] = np.degrees(np.arctan2(np.sin(outtake.angle), np.cos(outtake.angle)))
        current[index] = outtake.current
        force[index] = outtake.force
        torque[index] = outtake.torque
        angular_velocity[index] = np.degrees(outtake.angular_velocity)

        outtake.time_integrate(130E-4, 130E-4, timestep)
        current_time += timestep

    plt.figure()
    # plt.plot(time, angular_vel_real - angular_vel_pred, 'g', label="error")
    plt.plot(time, height, 'r', label="Height")
    plt.plot(time, angle, 'b', label="Angle")
    plt.plot(time, (angular_velocity), 'r', label="Angular velocity")
    plt.xlabel('time')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(time, torque, 'y', label="torque")
    plt.plot(time, force, 'pink', label="force")
    plt.xlabel('time')
    plt.legend()
    plt.show()
