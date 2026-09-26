/*
 * Optimal control loci of synchronous machines, ported from motulator.
 *
 * Python counterpart:
 *   motulator/drive/utils/_sm_control_loci.py  ControlLoci
 *
 * Both SynchronousMachinePars and SaturatedSynchronousMachinePars (with a flux
 * map) are supported, see sm_parameters.h. The loci are computed at NUM_LOCUS
 * points and stored as lookup tables, as in ReferenceGenerator of motulator.
 */

#ifndef MOTULATOR_SM_CONTROL_LOCI_H
#define MOTULATOR_SM_CONTROL_LOCI_H

#include "common.h"
#include "sm_parameters.h"

#define NUM_LOCUS 16

static double compute_mtpa_current_angle(const SynchronousMachinePars *par,
                                         double i_s_abs);
static double compute_mtpv_flux_angle(const SynchronousMachinePars *par,
                                      double psi_s_abs, double complex *i_s);
static double complex compute_mtpv_current(const SynchronousMachinePars *par,
                                           double i_s_abs);

/* Lookup tables of the ReferenceGenerator */
typedef struct {
    LUT psi_s_mtpa; /* MTPA flux magnitude vs. torque */
    LUT tau_M_cl;   /* Current-limit torque vs. flux magnitude */
    LUT tau_M_mtpv; /* MTPV torque vs. flux magnitude */
} ReferenceLUTs;

static void compute_reference_luts(const SynchronousMachinePars *par, double i_s_max,
                                   ReferenceLUTs *luts);

#endif
