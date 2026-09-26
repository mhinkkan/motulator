/*
 * Continuous-time model of a synchronous machine drive, ported from motulator. See
 * sm_machine.h.
 */

#include "sm_machine.h"

/* Secant method as scipy.optimize.newton without the derivative (x0 = 0) */
static double pm_flux_secant(const SpatialSaturatedSynchronousMachinePars *par)
{
    const double tol = 1.48e-8;
    double p0 = 0.0, p1 = 1e-4;
    double complex i_s;
    double tau;
    gradnet_current_map_harmonics(&par->current_map, par->k, p0, 0.0, &i_s, &tau);
    double q0 = creal(i_s);
    gradnet_current_map_harmonics(&par->current_map, par->k, p1, 0.0, &i_s, &tau);
    double q1 = creal(i_s);
    if (fabs(q1) < fabs(q0)) {
        double p_tmp = p0, q_tmp = q0;
        p0 = p1;
        q0 = q1;
        p1 = p_tmp;
        q1 = q_tmp;
    }
    for (int it = 0; it < 50; it++) {
        if (q1 == q0) {
            return 0.5 * (p1 + p0);
        }
        double p;
        if (fabs(q1) > fabs(q0)) {
            p = (-q0 / q1 * p1 + p0) / (1.0 - q0 / q1);
        } else {
            p = (-q1 / q0 * p0 + p1) / (1.0 - q1 / q0);
        }
        if (fabs(p - p1) <= tol) { /* numpy.isclose(p, p1, rtol=0, atol=tol) */
            return p;
        }
        p0 = p1;
        q0 = q1;
        p1 = p;
        gradnet_current_map_harmonics(&par->current_map, par->k, p1, 0.0, &i_s, &tau);
        q1 = creal(i_s);
    }
    return p1;
}

static SpatialSaturatedSynchronousMachinePars spatial_saturated_synchronous_machine_pars(
    double n_p, double R_s, const GradNet *current_map, int k)
{
    SpatialSaturatedSynchronousMachinePars par;
    par.n_p = n_p;
    par.R_s = R_s;
    par.current_map = *current_map;
    par.k = k;
    par.psi_f = pm_flux_secant(&par);
    if (par.psi_f < 1e-3) {
        par.psi_f = 0.0; /* No permanent magnets */
    }
    return par;
}

static void machine_magnetic_map(const SpatialSaturatedSynchronousMachinePars *par,
                                 double complex psi_s_dq, double theta_m,
                                 double complex *i_s_dq, double *tau_M)
{
    double tau_m;
    gradnet_current_map_harmonics(&par->current_map, par->k, psi_s_dq, theta_m, i_s_dq,
                                  &tau_m);
    *tau_M = par->n_p * tau_m;
}

static void machine_rhs(const SpatialSaturatedSynchronousMachinePars *par, double J,
                        double complex u_s_ab, double tau_L, double complex psi_s_dq,
                        double theta_M, double w_M, double complex *d_psi_s_dq,
                        double *d_theta_M, double *d_w_M)
{
    double theta_m = par->n_p * theta_M;
    double w_m = par->n_p * w_M;
    double complex i_s_dq;
    double tau_M;
    machine_magnetic_map(par, psi_s_dq, theta_m, &i_s_dq, &tau_M);

    /* Machine in rotor coordinates */
    double complex u_s_dq = u_s_ab * cexp(-I * theta_m);
    *d_psi_s_dq = u_s_dq - par->R_s * i_s_dq - I * w_m * psi_s_dq;

    /* Mechanical system */
    *d_theta_M = w_M;
    *d_w_M = (tau_M - tau_L) / J;
}
