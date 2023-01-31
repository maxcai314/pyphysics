cmake_minimum_required(VERSION 3.0)
project(MecanumMPC)
set(CMAKE_OSX_ARCHITECTURES "x86_64")

add_executable(${PROJECT_NAME} run_solver.c mecanum_mpc/include/mecanum_mpc.h)

target_include_directories(${PROJECT_NAME} PUBLIC
        mecanum_mpc/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        mecanum_mpc solver_with_model
)

target_link_directories(${PROJECT_NAME} PUBLIC
        mecanum_mpc/lib mecanum_mpc
)