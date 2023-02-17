from Mecanum_Drive.drive_log_reader import DriveLogReader

simulation = DriveLogReader('drive_samples/mpc_log_1.csv')
simulation.simulate_from_logs(time_step=1E-2, angle_wrap=False,relative_vel=True)
simulation.plot_evolution(legends = True)
simulation.plot_trajectory()
simulation.plot_acceleration()