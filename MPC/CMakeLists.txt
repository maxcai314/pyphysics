cmake_minimum_required(VERSION 3.0)
project(MecanumMPC)
set(CMAKE_OSX_ARCHITECTURES "x86_64")

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CPP_COMPILER gcc)

# add asan
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-omit-frame-pointer  -static -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -O2 -mno-sse -static")

add_executable(${PROJECT_NAME} run_solver.c ${CMAKE_CURRENT_SOURCE_DIR}/mecanum_mpc/include/mecanum_mpc.h  ${CMAKE_CURRENT_SOURCE_DIR}/mecanum_mpc/mecanum_mpc_model.c mecanum_mpc/mecanum_mpc_interface.c)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/mecanum_mpc/include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
        mecanum_mpc m
)

target_link_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/mecanum_mpc/lib ${CMAKE_CURRENT_SOURCE_DIR}/mecanum_mpc
)
