from idaapi import *
from ida_ua import *
from ida_bytes import *
from ida_typeinf import *
from ida_kernwin import *
from pyperclip import paste
import re
import tempfile

output_directory = '/Users/alec/kuriosity-power-play/mpc/src/main/cpp'

solver_typedefs = """
#define solver_int32_default int32_t

typedef double mecanum_mpc_float;
typedef double mecanum_mpc_ldl_s_float;
typedef double mecanum_mpc_ldl_r_float;
typedef double mecanum_mpc_callback_float;

typedef double mecanum_mpcinterface_float;
typedef struct mecanum_mpc_mem mecanum_mpc_mem;

typedef solver_int32_default (*mecanum_mpc_extfunc)(mecanum_mpc_float* x, mecanum_mpc_float* y, mecanum_mpc_float* lambda, mecanum_mpc_float* params, mecanum_mpc_float* pobj, mecanum_mpc_float* g, mecanum_mpc_float* c, mecanum_mpc_float* Jeq, mecanum_mpc_float* h, mecanum_mpc_float* Jineq, mecanum_mpc_float* H, solver_int32_default stage, solver_int32_default iterations, solver_int32_default threadID);


struct drive_parameters {
    double motor_constant;
    double armature_resistance;

    double robot_mass;
    double robot_moment;
    double wheel_moment;
    double roller_moment;

    double fl_wheel_friction;
    double fr_wheel_friction;
    double bl_wheel_friction;
    double br_wheel_friction;

    double fl_roller_friction;
    double fr_roller_friction;
    double bl_roller_friction;
    double br_roller_friction;

    double battery_voltage;
};


struct robot_target_state {
    double position[3];
    double velocity[3];
};

struct objective_weights {
    double motor_weights[4];
    double robot_position_weights[3];
    double robot_velocity_weights[3];
};

struct optimisation_parameters {
    struct drive_parameters model;
    struct robot_target_state target;
    struct objective_weights weights;
};

typedef struct drive_parameters drive_parameters;
typedef struct robot_target_state robot_target_state;
typedef struct objective_weights objective_weights;
typedef struct optimisation_parameters optimisation_parameters;


/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 100 */
    mecanum_mpc_float x0[100];

    /* vector of size 6 */
    mecanum_mpc_float xinit[6];

    /* vector of size 310 */
    struct optimisation_parameters all_parameters[10];


} mecanum_mpc_params;


typedef double robot_command[4];
struct robot_state {
    robot_command command;
    double position[3];
    double velocity[3];
};

/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* column vector of length 10 */
    struct robot_state x01;

    /* column vector of length 10 */
    struct robot_state x02;

    /* column vector of length 10 */
    struct robot_state x03;

    /* column vector of length 10 */
    struct robot_state x04;

    /* column vector of length 10 */
    struct robot_state x05;

    /* column vector of length 10 */
    struct robot_state x06;

    /* column vector of length 10 */
    struct robot_state x07;

    /* column vector of length 10 */
    struct robot_state x08;

    /* column vector of length 10 */
    struct robot_state x09;

    /* column vector of length 10 */
    struct robot_state x10;


} mecanum_mpc_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* scalar: iteration number */
    solver_int32_default it;

    /* scalar: number of iterations needed to optimality (branch-and-bound) */
    solver_int32_default it2opt;

    /* scalar: inf-norm of equality constraint residuals */
    mecanum_mpc_float res_eq;

    /* scalar: inf-norm of inequality constraint residuals */
    mecanum_mpc_float res_ineq;

    /* scalar: norm of stationarity condition */
    mecanum_mpc_float rsnorm;

    /* scalar: max of all complementarity violations */
    mecanum_mpc_float rcompnorm;

    /* scalar: primal objective */
    mecanum_mpc_float pobj;

    /* scalar: dual objective */
    mecanum_mpc_float dobj;

    /* scalar: duality gap := pobj - dobj */
    mecanum_mpc_float dgap;

    /* scalar: relative duality gap := |dgap / pobj | */
    mecanum_mpc_float rdgap;

    /* scalar: duality measure */
    mecanum_mpc_float mu;

    /* scalar: duality measure (after affine step) */
    mecanum_mpc_float mu_aff;

    /* scalar: centering parameter */
    mecanum_mpc_float sigma;

    /* scalar: number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;

    /* scalar: number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;

    /* scalar: step size (affine direction) */
    mecanum_mpc_float step_aff;

    /* scalar: step size (combined direction) */
    mecanum_mpc_float step_cc;

    /* scalar: total solve time */
    mecanum_mpc_float solvetime;

    /* scalar: time spent in function evaluations */
    mecanum_mpc_float fevalstime;

    /* column vector of length 8: solver ID of FORCESPRO solver */
    solver_int32_default solver_id[8];




} mecanum_mpc_info;
"""

# save solver typedefs to a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.h', delete=False) as f:
    f.write(solver_typedefs)
    f.flush()
    f.close()

    # load the temporary file as a header file
    idc_parse_types(f.name, PT_FILE)

method_defs = """
int mecanum_mpc_solve(
        mecanum_mpc_params *params,
        mecanum_mpc_output *output,
        mecanum_mpc_info *info,
        mecanum_mpc_mem *mem,
        FILE *fs,
        mecanum_mpc_extfunc evalextfunctions_mecanum_mpc);
double mecanum_mpc_toc(struct timespec (*a1)[2]);
int mecanum_mpc_tic(struct timespec *a1);
int linesearch_check_filter_mecanum_mpc(double *a1, double a2, double a3);
void computePukN(double *a1, double *a2, double *a3, double *a4, double *a5, double *a6, double a7);
"""
for method_def in method_defs.split(';'):
    if not method_def.strip():
        continue

    method_def = method_def.replace('\n', ' ').replace('\r', '').replace('\t', ' ').replace('\s+', ' ') + ';'
    method_name = re.search(r'(\w+)\(', method_def).group(1)
    method_ea = get_name_ea_simple(method_name)
    if method_ea == BADADDR:
        print(f'Could not find method {method_name}')
        continue

    print(method_def)
    apply_cdecl(idaapi.cvar.idati, method_ea, method_def)

solver_typedefs += method_defs

# for all functins that begin with "f_", or f then a number, change the first argument to be a double pointer
double_ptr = tinfo_t()
double_ptr.create_ptr(tinfo_t(BTF_DOUBLE))

stack_chk_fail = list(XrefsTo(get_name_ea_simple("__stack_chk_fail")))[0].frm

def fix_argument_types(ea):
    activate_widget(find_widget('Pseudocode-A'), True)
    jumpto(ea)

    tif = tinfo_t()
    funcdata = func_type_data_t()
    get_tinfo(tif, ea)
    tif.get_func_details(funcdata)

    new_funcdata = func_type_data_t()
    tif.get_func_details(new_funcdata)

    for pos, arg in enumerate(funcdata):
        if str(arg.type) == "__int64" or str(arg.type) == "_QWORD":
            print(f"setting {pos} in {ida_funcs.get_func_name(ea)} to double*")
            new_funcdata[pos].type = double_ptr

    new_funcdata.rettype = tinfo_t(BTF_VOID)
    # set cc to cdecl
    new_funcdata.cc = CM_CC_CDECL

    new_tif = tinfo_t()
    new_tif.create_func(new_funcdata)
    apply_tinfo(ea, new_tif, TINFO_DEFINITE)

for ea in Functions():
    if re.match(r"f[_0-9]+", get_func_name(ea)) or re.match(r"la_init_pushback_.+", get_func_name(ea)):
        pass
        #fix_argument_types(ea)

# remove licence checks
#is_licensed_mecanum_mpc = get_name_ea_simple("is_licensed_mecanum_mpc")
#for xref in XrefsTo(is_licensed_mecanum_mpc):
#    if not get_bytes(xref.frm, 5) == b"\xE8\xEA\x8E\xFF\xFF":
#        continue

#    # replace the call instruction with the mov instruction
#    patch_bytes(xref.frm, b"\xB8\x01\x00\x00\x00")

matrix_functions_c = """
#include <stdint.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#include <stdio.h>

#include "matrix_functions.h"
#include "locs.h"
"""

matrix_functions_h = """
#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H

#include <stdint.h>
#include "locs.h"
#define __time_t time_t
#define __syscall_slong_t long int

#define __readfsqword(x) 0

#define LODWORD(x) ((uint32_t)(x))

#define _OWORD uint128_t
#define _QWORD uint64_t
#define _DWORD uint32_t
#define _BYTE uint8_t

#define __int64 long long int
#define __int32 int
#define __int16 short int
#define __int8 char

#define FMIN_FORCES fmin
#define FMAX_FORCES fmax
#define IMIN_FORCES fmin
#define IMAX_FORCES fmax

#include "mecanum_mpc.h"
"""

widget = find_widget('Pseudocode-A')

previous_method_name = None
for ea in Functions():
    enable = ['rowToColumnMajorMixedIndices', 'diffU', 'diffX', 'integrationSum', 'evaluateCallbacks', 'addIdToSquareDenseMat', 'computPxkN', 'sparseToFullJacX', 'matMulDimxDimx', 'computePukN', 'sparseToFullJacU', 'denseRowMajPlusCcsTransDimuDimx', 'matMulDimxDimu', 'addMultiple', 'mecanum_mpc_internal_mem']
    if re.match(r"f[_0-9]+", get_func_name(ea)) or re.match(r"la_.+", get_func_name(ea)) or re.match(r"linesearch.+", get_func_name(ea)) or re.match(r"mecanum_mpc_(?!solve|internal)", get_func_name(ea)) or get_func_name(ea) in enable:
        activate_widget(widget, True)

        copied_name = previous_method_name
        while copied_name is None or copied_name == previous_method_name:
            jumpto(ea)
            process_ui_action('SelectAll')
            process_ui_action('EditCopy')
            code = paste()
            if not code:
                continue
            copied_name = code.split("{")[0].strip()

        previous_method_name = copied_name

        matrix_functions_c += code
        matrix_functions_c += "\n\n"
        matrix_functions_h += code.split("{")[0].strip() + ";\n"

matrix_functions_h += "#endif\n"

with open(output_directory + "/matrix_functions.c", "w") as f:
    f.write(matrix_functions_c)

with open(output_directory + "/matrix_functions.h", "w") as f:
    f.write(matrix_functions_h)

mecanum_mpc_c = """
#include "mecanum_mpc.h"
#include "matrix_functions.h"

uint8_t** mecanum_mpc_mem_internal = {NULL, NULL};
uint8_t vars_2747[5][4112];
"""

mecanum_mpc_h = """
#ifndef MECANUM_MPC_H
#define MECANUM_MPC_H

#include <stdint.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <tgmath.h>
#include <time.h>
#include <stdlib.h>
#include "locs.h"

#define _OWORD uint128_t
#define _QWORD uint64_t
#define _DWORD uint32_t
#define _BYTE uint8_t
#define __readfsqword(x) 0
#define __int64 long long int


#define FMIN_FORCES fmin
#define FMAX_FORCES fmax
#define IMIN_FORCES fmin
#define IMAX_FORCES fmax

__int64 mecanum_mpc_get_mem_size(void);
""" + "\n" + solver_typedefs

widget = find_widget('Pseudocode-A')
mecanum_mpc_solve = get_name_ea_simple("mecanum_mpc_solve")
activate_widget(widget, True)
jumpto(mecanum_mpc_solve)
process_ui_action('SelectAll')
process_ui_action('EditCopy')
code = paste()

mecanum_mpc_c += code
mecanum_mpc_h += code.split("{")[0].strip() + ";\n"
mecanum_mpc_h += "#endif\n"

with open(output_directory + "/mecanum_mpc.c", "w") as f:
    f.write(mecanum_mpc_c)

with open(output_directory + "/mecanum_mpc.h", "w") as f:
    f.write(mecanum_mpc_h)