/*
 * Continuous-time model of a synchronous machine drive, ported from motulator.
 *
 * Python counterparts:
 *   motulator/drive/model/_machine.py      SynchronousMachine
 *   motulator/drive/model/_mechanics.py    MechanicalSystem
 *   motulator/drive/utils/_parameters.py   SpatialSaturatedSynchronousMachinePars
 *
 * The machine is modeled in rotor coordinates, and the stator flux linkage is the
 * state variable. The magnetic model is a GradNet current map with spatial
 * harmonics (CurrentMapWithHarmonics), giving the current and the torque as
 * functions of the flux linkage and the electrical rotor angle. The core losses are
 * not modeled (G_c = 0).
 */

#ifndef MOTULATOR_SM_MACHINE_H
#define MOTULATOR_SM_MACHINE_H

#include "gradnet.h"

/* Parameters of the machine (SpatialSaturatedSynchronousMachinePars) */
typedef struct {
    double n_p;
    double R_s;
    GradNet current_map; /* Current map with spatial harmonics */
    int k;               /* Spatial harmonic order */
    double psi_f;        /* PM-flux linkage, computed from the current map */
} SpatialSaturatedSynchronousMachinePars;

static SpatialSaturatedSynchronousMachinePars spatial_saturated_synchronous_machine_pars(
    double n_p, double R_s, const GradNet *current_map, int k);

/* Magnetic map: stator current (A) in rotor coordinates and electromagnetic torque
 * (Nm), as functions of the stator flux linkage (Vs) in rotor coordinates and the
 * electrical rotor angle (rad) */
static void machine_magnetic_map(const SpatialSaturatedSynchronousMachinePars *par,
                                 double complex psi_s_dq, double theta_m,
                                 double complex *i_s_dq, double *tau_M);

/* State derivatives of the machine and the mechanical system (MechanicalSystem
 * without friction) */
static void machine_rhs(const SpatialSaturatedSynchronousMachinePars *par, double J,
                        double complex u_s_ab, double tau_L, double complex psi_s_dq,
                        double theta_M, double w_M, double complex *d_psi_s_dq,
                        double *d_theta_M, double *d_w_M);

#endif
