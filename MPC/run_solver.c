//
// Created by Alec Petridis on 1/30/23.
//

#include <stdlib.h>
#include "mecanum_mpc.h"
#include "mecanum_mpc_memory.h"

#define NUMSTAGES 10
#define NUMVARS 10

extern mecanum_mpc_extfunc mecanum_mpc_adtool2forces;

struct __attribute__((__packed__)) objective_term {
    mecanum_mpc_float desired;
    mecanum_mpc_float weight;
};

struct __attribute__((__packed__)) mecanum_solver_params {
    mecanum_mpc_float battery_voltage;

    struct objective_term fl, fr, bl, br;
    struct objective_term x, y, theta;
    struct objective_term x_vel, y_vel, theta_vel;
};

int main() {
    mecanum_mpc_params *params = malloc(sizeof(mecanum_mpc_params));
    mecanum_mpc_output *output = malloc(sizeof(mecanum_mpc_output));
    mecanum_mpc_info *info = malloc(sizeof(mecanum_mpc_info));
    mecanum_mpc_mem *args = mecanum_mpc_internal_mem(0);
    FILE *debug_output = stdout;

    memset(params, 0, sizeof(*params));
    for (int i = 0; i < NUMSTAGES; i++) {
        struct mecanum_solver_params *stage_params = &((struct mecanum_solver_params *) (params->all_parameters))[i];

        stage_params->battery_voltage = 12.0;
        stage_params->x.desired = 1;
        stage_params->y.desired = 0;

        stage_params->x.weight = 1;
        stage_params->y.weight = 1;
    }

    mecanum_mpc_solve((mecanum_mpc_params *) params, output, info, args, debug_output, &mecanum_mpc_adtool2forces);
    printf("pobj: %f", info->pobj);
    // print output
    for (int i = 0; i < NUMSTAGES; i++) {
        printf("Stage %d: ", i);
        for (int j = 0; j < NUMSTAGES; j++) {
            printf("%f ", *(((mecanum_mpc_float *) output) + i * NUMVARS + j));
        }
        printf("\n");
    }
}