/*
 * Synchronous machine model parameters, ported from motulator. See sm_parameters.h.
 */

#include "sm_parameters.h"

#define EPS 1e-3 /* As EPS in motulator/drive/utils/_parameters.py */

static SynchronousMachinePars synchronous_machine_pars(double n_p, double R_s,
                                                       double L_d, double L_q,
                                                       double psi_f)
{
    SynchronousMachinePars par = {0};
    par.n_p = n_p;
    par.R_s = R_s;
    par.L_d = L_d;
    par.L_q = L_q;
    par.psi_f = psi_f;
    par.saturated = 0;
    return par;
}

static SynchronousMachinePars saturated_synchronous_machine_pars(
    double n_p, double R_s, const GradNet *flux_map)
{
    SynchronousMachinePars par = {0};
    par.n_p = n_p;
    par.R_s = R_s;
    par.saturated = 1;
    par.flux_map = *flux_map;
    /* As in SaturatedSynchronousMachinePars.__post_init__ */
    par.psi_f = creal(psi_s_dq(&par, 0.0));
    IncrIndMat L0 = incr_ind_mat(&par, 0.0);
    par.L_d0 = L0.L_dd;
    par.L_q0 = L0.L_qq;
    if (par.psi_f < EPS) {
        par.psi_f = 0.0; /* No permanent magnets */
    }
    return par;
}

static double complex psi_s_dq(const SynchronousMachinePars *par, double complex i_s)
{
    if (par->saturated) {
        return gradnet_map(&par->flux_map, i_s);
    }
    return par->L_d * creal(i_s) + par->psi_f + I * par->L_q * cimag(i_s);
}

static IncrIndMat incr_ind_mat(const SynchronousMachinePars *par, double complex i_s)
{
    IncrIndMat L;
    if (!par->saturated) {
        L.L_dd = par->L_d;
        L.L_dq = 0.0;
        L.L_qq = par->L_q;
        return L;
    }
    /* Central differences */
    double complex psi_dev_d = psi_s_dq(par, i_s + EPS) - psi_s_dq(par, i_s - EPS);
    double complex psi_dev_q =
        psi_s_dq(par, i_s + I * EPS) - psi_s_dq(par, i_s - I * EPS);
    L.L_dd = creal(psi_dev_d) / (2 * EPS);
    L.L_qq = cimag(psi_dev_q) / (2 * EPS);
    L.L_dq = creal(psi_dev_q) / (2 * EPS);
    return L;
}

static double complex iterate_i_s_dq(const SynchronousMachinePars *par,
                                     double complex psi_s)
{
    if (!par->saturated) {
        return (creal(psi_s) - par->psi_f) / par->L_d + I * cimag(psi_s) / par->L_q;
    }
    /* Newton's method with the incremental inductance matrix as the Jacobian (the
     * root is the same as with scipy.optimize.root in motulator) */
    double complex i_s = (creal(psi_s) - par->psi_f) / par->L_d0
                         + I * cimag(psi_s) / par->L_q0;
    for (int k = 0; k < 50; k++) {
        double complex err = psi_s_dq(par, i_s) - psi_s;
        if (cabs(err) < 1e-12 * (1.0 + cabs(psi_s))) {
            break;
        }
        IncrIndMat L = incr_ind_mat(par, i_s);
        double det_L = L.L_dd * L.L_qq - L.L_dq * L.L_dq;
        double d_i_d = (L.L_qq * creal(err) - L.L_dq * cimag(err)) / det_L;
        double d_i_q = (L.L_dd * cimag(err) - L.L_dq * creal(err)) / det_L;
        i_s -= d_i_d + I * d_i_q;
    }
    return i_s;
}

static double complex aux_flux(const IncrIndMat *L, double complex i_s,
                               double complex psi_s)
{
    return psi_s - L->L_qq * creal(i_s) - I * L->L_dd * cimag(i_s)
           + I * L->L_dq * conj(i_s);
}

static double complex aux_current(const IncrIndMat *L, double complex psi_s,
                                  double complex i_s)
{
    double det_L = L->L_dd * L->L_qq - L->L_dq * L->L_dq;
    return (L->L_dd * creal(psi_s) + I * L->L_qq * cimag(psi_s)
            + I * L->L_dq * conj(psi_s))
               / det_L
           - i_s;
}
