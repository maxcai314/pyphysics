//
// Created by Alec Petridis on 1/30/23.
//

#include <stdlib.h>
#include "mecanum_mpc.h"
#include "mecanum_mpc_memory.h"

#include "musl_compat.h"

#define NUMSTAGES 5
#define NUMVARS 10

extern mecanum_mpc_extfunc mecanum_mpc_adtool2forces;

void print_mecanum_mpc_info(mecanum_mpc_info *info) {
    // print all solver info besides the solver ID in a format easily parsable on a single line
    printf("it: %d, it2opt: %d, res_eq: %e, res_ineq: %e, rsnorm: %e, rcompnorm: %e, pobj: %e, dobj: %e, dgap: %e, rdgap: %e, mu: %e, mu_aff: %e, sigma: %e, lsit_aff: %d, lsit_cc: %d, step_aff: %e, step_cc: %e, solvetime: %e, fevalstime: %e",
           info->it, info->it2opt, info->res_eq, info->res_ineq, info->rsnorm, info->rcompnorm, info->pobj, info->dobj,
           info->dgap, info->rdgap, info->mu, info->mu_aff, info->sigma, info->lsit_aff, info->lsit_cc, info->step_aff,
           info->step_cc, info->solvetime, info->fevalstime);
}

void print_mecanum_mpc_output(mecanum_mpc_output *output) {
    mecanum_mpc_float *output_vars = (mecanum_mpc_float *) output;
    for (int i = 0; i < NUMSTAGES; i++) {
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

    for (int i = 0; i < (sizeof(*params) / sizeof(mecanum_mpc_float)); i++) {
        scanf("%lf", ((mecanum_mpc_float *) params) + i);
    }

    mecanum_mpc_solve((mecanum_mpc_params *) params, output, info, args, debug_output, &mecanum_mpc_adtool2forces);

    print_mecanum_mpc_info(info);
    print_mecanum_mpc_output(output);

    free(params);
    free(output);
    free(info);
}
