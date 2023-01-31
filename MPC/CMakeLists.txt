cmake_minimum_required(VERSION 3.0)
project(MecanumMPC)

add_executable(${PROJECT_NAME} run_solver.c mecanum_mpc/include/mecanum_mpc.h)

target_include_directories(${PROJECT_NAME} PUBLIC
        mecanum_mpc/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        mecanum_mpc
)

target_link_directories(${PROJECT_NAME} PUBLIC
        mecanum_mpc/lib
)