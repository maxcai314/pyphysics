//
// Created by Alec Petridis on 1/30/23.
//

#include <stdlib.h>
#include "mecanum_mpc.h"
#include "mecanum_mpc_memory.h"

extern mecanum_mpc_extfunc mecanum_mpc_adtool2forces;

int main() {
    mecanum_mpc_params* params = malloc(sizeof(mecanum_mpc_params));
    mecanum_mpc_output* output = malloc(sizeof(mecanum_mpc_output));
    mecanum_mpc_info* info = malloc(sizeof(mecanum_mpc_info));
    mecanum_mpc_mem* args = mecanum_mpc_internal_mem(0);
    FILE* debug_output = stdout;

    memset(&params->x0, 0, sizeof(params->x0));
    mecanum_mpc_float (* starting_position)[3] = &params->xinit;
}