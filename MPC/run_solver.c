//
// Created by Alec Petridis on 1/30/23.
//

#include <stdlib.h>
#include "mecanum_mpc.h"
#include "mecanum_mpc_memory.h"

#include "musl_compat.h"

#define NUMSTAGES 5
#define NUMVARS 10
#define NUMPARAMS 21

extern mecanum_mpc_extfunc mecanum_mpc_adtool2forces;

void print_mecanum_mpc_info(mecanum_mpc_info *info) {
    // print all solver info besides the solver ID in a format easily parsable on a single line
    printf("INFO %d %d %e %e %e %e %e %e %e %e %e %e %e %d %d %e %e %e %e\n",
           info->it, info->it2opt, info->res_eq, info->res_ineq, info->rsnorm, info->rcompnorm, info->pobj, info->dobj,
           info->dgap, info->rdgap, info->mu, info->mu_aff, info->sigma, info->lsit_aff, info->lsit_cc, info->step_aff,
           info->step_cc, info->solvetime, info->fevalstime);
}

void print_mecanum_mpc_output(mecanum_mpc_output *output) {
    mecanum_mpc_float *output_vars = (mecanum_mpc_float *) output;
    for (int i = 0; i < NUMSTAGES; i++) {
        printf("STAGE ");
        for (int j = 0; j < NUMVARS; j++) {
            printf("%e ", output_vars[i * NUMVARS + j]);
        }
        printf("\n");
    }
}

int main() {
    mecanum_mpc_params *params = malloc(sizeof(mecanum_mpc_params));
    mecanum_mpc_output *output = malloc(sizeof(mecanum_mpc_output));
    mecanum_mpc_info *info = malloc(sizeof(mecanum_mpc_info));
    mecanum_mpc_mem *args = mecanum_mpc_internal_mem(0);
    FILE *debug_output = stdout;

    for (int i = 0; i < NUMSTAGES; i++) {
        int result = scanf(" INITIAL %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
              &((mecanum_mpc_float *) params)[i * NUMVARS + 0],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 1],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 2],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 3],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 4],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 5],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 6],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 7],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 8],
              &((mecanum_mpc_float *) params)[i * NUMVARS + 9]);
        if (result != 10) {
            printf("Error reading initial conditions for stage %d\n", i);
            goto end;
        }
    }
    int result = scanf(" XINIT %lf %lf %lf %lf %lf %lf", &params->xinit[0], &params->xinit[1], &params->xinit[2], &params->xinit[3],
          &params->xinit[4], &params->xinit[5]);

    if (result != 6) {
        printf("Error reading xinit\n");
        goto end;
    }
    for (int i = 0; i < NUMSTAGES; i++) {
        printf("Reading stage %d\n", i);
        int result = scanf(" STAGE %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                           &( params->all_parameters)[i * NUMPARAMS + 0],
                           &params->all_parameters[i * NUMPARAMS + 1],
                           &params->all_parameters[i * NUMPARAMS + 2],
                           &params->all_parameters[i * NUMPARAMS + 3],
                           &params->all_parameters[i * NUMPARAMS + 4],
                           &params->all_parameters[i * NUMPARAMS + 5],
                           &params->all_parameters[i * NUMPARAMS + 6],
                           &params->all_parameters[i * NUMPARAMS + 7],
                           &params->all_parameters[i * NUMPARAMS + 8],
                           &params->all_parameters[i * NUMPARAMS + 9],
                           &params->all_parameters[i * NUMPARAMS + 10],
                           &params->all_parameters[i * NUMPARAMS + 11],
                           &params->all_parameters[i * NUMPARAMS + 12],
                           &params->all_parameters[i * NUMPARAMS + 13],
                           &params->all_parameters[i * NUMPARAMS + 14],
                           &params->all_parameters[i * NUMPARAMS + 15],
                           &params->all_parameters[i * NUMPARAMS + 16],
                           &params->all_parameters[i * NUMPARAMS + 17],
                           &params->all_parameters[i * NUMPARAMS + 18],
                           &params->all_parameters[i * NUMPARAMS + 19],
                           &params->all_parameters[i * NUMPARAMS + 20]);
        if (result != 21) {
            printf("Error reading stage %d", i);
            goto end;
        }
    }

    mecanum_mpc_solve((mecanum_mpc_params *) params, output, info, args, debug_output, &mecanum_mpc_adtool2forces);

    printf("BEGIN\n");
    print_mecanum_mpc_info(info);
    print_mecanum_mpc_output(output);

    end:
    free(params);
    free(output);
    free(info);
}
