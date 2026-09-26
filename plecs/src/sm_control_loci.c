/*
 * Optimal control loci of synchronous machines, ported from motulator. See
 * sm_control_loci.h.
 */

#include "sm_control_loci.h"

/* Machine model (constant inductances) ------------------------------------- */

static double complex psi_s_dq(const SynchronousMachinePars *par, double complex i_s)
{
    return par->L_d * creal(i_s) + par->psi_f + I * par->L_q * cimag(i_s);
}

static double complex i_s_dq(const SynchronousMachinePars *par, double complex psi_s)
{
    return (creal(psi_s) - par->psi_f) / par->L_d + I * cimag(psi_s) / par->L_q;
}

static double complex aux_flux(const SynchronousMachinePars *par, double complex i_s,
                               double complex psi_s)
{
    double L_dd = par->L_d, L_qq = par->L_q, L_dq = 0.0;
    return psi_s - L_qq * creal(i_s) - I * L_dd * cimag(i_s) + I * L_dq * conj(i_s);
}

static double complex aux_current(const SynchronousMachinePars *par,
                                  double complex psi_s, double complex i_s)
{
    double L_dd = par->L_d, L_qq = par->L_q, L_dq = 0.0;
    double det_L = L_dd * L_qq - L_dq * L_dq;
    return (L_dd * creal(psi_s) + I * L_qq * cimag(psi_s) + I * L_dq * conj(psi_s))
               / det_L
           - i_s;
}

static double linspace(double start, double stop, int num, int k)
{
    /* Element k of numpy.linspace(start, stop, num) */
    if (k == num - 1) {
        return stop;
    }
    return k * ((stop - start) / (num - 1)) + start;
}

/* Conditions for the root finding ------------------------------------------- */

typedef struct {
    const SynchronousMachinePars *par;
    double magnitude;
    double complex i_s; /* Current at the latest evaluation (side effect) */
} LocusData;

static double mtpa_cond(double gamma, void *data)
{
    LocusData *d = (LocusData *)data;
    double complex i_s = d->magnitude * cexp(I * gamma);
    double complex psi_s = psi_s_dq(d->par, i_s);
    double complex psi_a = aux_flux(d->par, i_s, psi_s);
    return creal(psi_a * conj(i_s));
}

static double mtpv_flux_cond(double delta, void *data)
{
    LocusData *d = (LocusData *)data;
    double complex psi_s = d->magnitude * cexp(I * delta);
    d->i_s = i_s_dq(d->par, psi_s);
    double complex i_a = aux_current(d->par, psi_s, d->i_s);
    return creal(i_a * conj(psi_s));
}

static double mtpv_current_cond(double gamma, void *data)
{
    LocusData *d = (LocusData *)data;
    double complex i_s = d->magnitude * cexp(I * gamma);
    double complex psi_s = psi_s_dq(d->par, i_s);
    double complex i_a = aux_current(d->par, psi_s, i_s);
    return creal(i_a * conj(psi_s));
}

static void angle_range(const SynchronousMachinePars *par, double range[2])
{
    range[0] = (par->psi_f == 0.0) ? 0.0 : 0.5 * M_PI;
    range[1] = (par->psi_f == 0.0) ? 0.5 * M_PI : M_PI;
}

/* ControlLoci -------------------------------------------------------------- */

static double compute_mtpa_current_angle(const SynchronousMachinePars *par,
                                         double i_s_abs)
{
    LocusData d = {par, i_s_abs, 0.0};
    double r[2];
    angle_range(par, r);
    if (mtpa_cond(r[0], &d) * mtpa_cond(r[1], &d) > 0.0) {
        return 0.0; /* No root in the range */
    }
    return brentq(mtpa_cond, r[0], r[1], &d);
}

static double compute_mtpv_flux_angle(const SynchronousMachinePars *par,
                                      double psi_s_abs, double complex *i_s)
{
    LocusData d = {par, psi_s_abs, NAN};
    double r[2];
    angle_range(par, r);
    if (mtpv_flux_cond(r[0], &d) * mtpv_flux_cond(r[1], &d) > 0.0) {
        *i_s = NAN;
        return NAN; /* No root in the range */
    }
    double delta = brentq(mtpv_flux_cond, r[0], r[1], &d);
    *i_s = d.i_s; /* Current at the latest evaluation, as in motulator */
    return delta;
}

static double complex compute_mtpv_current(const SynchronousMachinePars *par,
                                           double i_s_abs)
{
    LocusData d = {par, i_s_abs, 0.0};
    double r[2];
    angle_range(par, r);
    if (mtpv_current_cond(r[0], &d) * mtpv_current_cond(r[1], &d) >= 0.0) {
        return NAN; /* No MTPV for this current */
    }
    double gamma = brentq(mtpv_current_cond, r[0], r[1], &d);
    return i_s_abs * cexp(I * gamma);
}

/* Lookup tables as in ReferenceGenerator ------------------------------------ */

static double torque(const SynchronousMachinePars *par, double complex i_s,
                     double complex psi_s)
{
    return 1.5 * par->n_p * cimag(i_s * conj(psi_s));
}

static void compute_reference_luts(const SynchronousMachinePars *par, double i_s_max,
                                   ReferenceLUTs *luts)
{
    int num = NUM_LOCUS;
    double complex i_s, psi_s, i_s_mtpa_max = 0.0, psi_s_mtpa_max = 0.0;

    /* MTPA locus */
    luts->psi_s_mtpa.n = num;
    for (int k = 0; k < num; k++) {
        double i_s_abs = linspace(0.0, i_s_max, num, k);
        double gamma = compute_mtpa_current_angle(par, i_s_abs);
        i_s = i_s_abs * cexp(I * gamma);
        psi_s = psi_s_dq(par, i_s);
        luts->psi_s_mtpa.x[k] = torque(par, i_s, psi_s);
        luts->psi_s_mtpa.y[k] = cabs(psi_s);
        i_s_mtpa_max = i_s;
        psi_s_mtpa_max = psi_s;
    }

    /* MTPV limit */
    luts->tau_M_mtpv.n = num;
    for (int k = 0; k < num; k++) {
        double psi_s_abs = linspace(0.0, cabs(psi_s_mtpa_max), num, k);
        double delta = compute_mtpv_flux_angle(par, psi_s_abs, &i_s);
        psi_s = psi_s_abs * cexp(I * delta);
        luts->tau_M_mtpv.x[k] = cabs(psi_s);
        luts->tau_M_mtpv.y[k] = torque(par, i_s, psi_s);
    }

    /* Current limit, from the MTPV (or pi, if no MTPV) to the MTPA current angle */
    double gamma1 = carg(compute_mtpv_current(par, i_s_max));
    double gamma2 = carg(i_s_mtpa_max);
    if (isnan(gamma1)) {
        gamma1 = M_PI;
    }
    luts->tau_M_cl.n = num;
    for (int k = 0; k < num; k++) {
        double gamma = linspace(gamma1, gamma2, num, k);
        i_s = i_s_max * cexp(I * gamma);
        psi_s = psi_s_dq(par, i_s);
        luts->tau_M_cl.x[k] = cabs(psi_s);
        luts->tau_M_cl.y[k] = torque(par, i_s, psi_s);
    }
}
