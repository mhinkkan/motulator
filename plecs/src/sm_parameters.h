/*
 * Synchronous machine model parameters, ported from motulator.
 *
 * Python counterpart:
 *   motulator/drive/utils/_parameters.py  SynchronousMachinePars,
 *                                         SaturatedSynchronousMachinePars
 *
 * The same structure represents both classes. If `saturated` is nonzero, the flux
 * linkage is given by a GradNet flux map (psi_s_dq_fcn in motulator), and the
 * incremental inductances are computed with finite differences, as in motulator.
 * Otherwise, the constant inductances L_d and L_q are used.
 */

#ifndef MOTULATOR_SM_PARAMETERS_H
#define MOTULATOR_SM_PARAMETERS_H

#include "gradnet.h"

typedef struct {
    double n_p;
    double R_s;
    double L_d; /* Constant inductances (SynchronousMachinePars) */
    double L_q;
    double psi_f;
    int saturated;    /* SaturatedSynchronousMachinePars with a flux map */
    GradNet flux_map; /* Flux linkage as a function of the current */
    double L_d0;      /* Incremental inductances at zero current */
    double L_q0;
} SynchronousMachinePars;

/* Incremental inductance matrix [[L_dd, L_dq], [L_dq, L_qq]] */
typedef struct {
    double L_dd;
    double L_dq;
    double L_qq;
} IncrIndMat;

static SynchronousMachinePars synchronous_machine_pars(double n_p, double R_s,
                                                       double L_d, double L_q,
                                                       double psi_f);
static SynchronousMachinePars saturated_synchronous_machine_pars(
    double n_p, double R_s, const GradNet *flux_map);

static double complex psi_s_dq(const SynchronousMachinePars *par, double complex i_s);
static IncrIndMat incr_ind_mat(const SynchronousMachinePars *par, double complex i_s);
static double complex iterate_i_s_dq(const SynchronousMachinePars *par,
                                     double complex psi_s);

/* Auxiliary flux linkage and current vectors */
static double complex aux_flux(const IncrIndMat *L, double complex i_s,
                               double complex psi_s);
static double complex aux_current(const IncrIndMat *L, double complex psi_s,
                                  double complex i_s);

#endif
