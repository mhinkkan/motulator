/*
 * Sensorless flux-vector control of synchronous machine drives, ported from
 * motulator.
 *
 * Python counterparts:
 *   motulator/drive/control/_sm_observers.py      FluxObserver, SpeedFluxObserver
 *   motulator/drive/control/_sm_reference_gen.py  ReferenceGenerator (LUT-based)
 *   motulator/drive/control/_sm_flux_vector.py    FluxTorqueController,
 *                                                 FluxVectorController
 *   motulator/drive/control/_base.py              VectorControlSystem
 *   motulator/drive/control/_common.py            SpeedController
 *
 * The machine model parameters correspond to SynchronousMachinePars, i.e., a
 * machine without magnetic saturation. The control loop follows motulator: the
 * output function is free of side effects (apart from workspace variables) and
 * all states are updated in the update function, in the same order as in
 * motulator.
 */

#ifndef MOTULATOR_SM_FLUX_VECTOR_H
#define MOTULATOR_SM_FLUX_VECTOR_H

#include "common.h"
#include "sm_control_loci.h"

/* Feedback signals for the control system (ObserverOutputs) */
typedef struct {
    double u_dc;           /* DC-bus voltage */
    double complex i_s;    /* Stator current */
    double complex u_s;    /* Stator voltage */
    double complex psi_s;  /* Stator flux linkage estimate */
    double complex e_o;    /* Flux estimation error signal */
    double eps;            /* Mechanical position estimation error signal */
    double eps_f;          /* PM-flux error signal */
    double complex psi_a;  /* Auxiliary flux linkage */
    double tau_M;          /* Electromagnetic torque estimate */
    double tau_L;          /* Load torque estimate */
    double w_c;            /* Angular speed of the coordinate system */
    double w_m;            /* Electrical angular rotor speed estimate */
    double w_M;            /* Mechanical angular rotor speed estimate */
    double theta_c;        /* Coordinate system angle */
    double theta_m;        /* Electrical rotor angle estimate */
    double psi_f;          /* PM-flux linkage estimate */
    double h;              /* Weight of the external position error signal */
} ObserverOutputs;

/* Flux observer in estimated rotor coordinates (sensorless mode, h = 0) */
typedef struct {
    SynchronousMachinePars par;
    double k_theta;
    double k_o[2]; /* Observer gain k_o(w_m) = k_o[0] + k_o[1]*abs(w_m) */
    /* States */
    double theta_m;
    double complex psi_s;
} FluxObserver;

typedef struct {
    SpeedObserver speed_observer;
    FluxObserver flux_observer;
} SpeedFluxObserver;

/* Flux and torque reference generator based on precomputed lookup tables */
typedef struct {
    double k_u;
    double k_mtpv;
    double psi_s_min;
    double psi_s_max;
    LUT psi_s_mtpa;  /* MTPA flux magnitude vs. torque */
    LUT tau_M_cl;    /* Current-limit torque vs. flux magnitude */
    LUT tau_M_mtpv;  /* MTPV torque vs. flux magnitude */
} ReferenceGenerator;

typedef struct {
    SynchronousMachinePars par;
    double alpha_psi;
    double alpha_tau;
    double alpha_i;
    /* Integral states */
    double x_psi;
    double x_tau;
    /* Workspace variables */
    double complex i_a;
    double complex v;
} FluxTorqueController;

typedef struct {
    double T_s;
    ReferenceGenerator reference_gen;
    FluxTorqueController flux_torque_ctrl;
    SpeedFluxObserver observer;
} FluxVectorController;

/* Reference signals */
typedef struct {
    double T_s;
    double d_abc[3];
    double w_M;
    double tau_M;
    double psi_s;
    double complex u_s;
    double complex u_c_ab; /* Limited converter voltage from the PWM */
} References;

/* Measured signals */
typedef struct {
    double complex i_c_ab; /* Converter current in stationary coordinates */
    double u_dc;
} Measurements;

typedef struct {
    PWM pwm;
    FluxVectorController vector_ctrl;
    PIController speed_ctrl;
    /* Workspace variables passed from the output to the update function */
    ObserverOutputs fbk;
    References ref;
} VectorControlSystem;

/* Flux-vector controller configuration (FluxVectorControllerCfg). As in motulator,
 * the optional parameters default to None, represented here by NAN. Only the
 * sensorless mode (sensorless=True) with offline reference generation
 * (online_ref=False) and the default PM-flux estimation gain (k_f=None) are
 * supported. The observer gain k_o is given as k_o(w_m) = k_o[0] + k_o[1]*abs(w_m). */
typedef struct {
    double i_s_max;
    double alpha_tau;
    double alpha_psi;
    double alpha_i;
    double alpha_o;
    double k_o[2];
    double psi_s_min;
    double psi_s_max;
    double k_u;
    double k_mtpv;
    double J;
    double T_s;
} FluxVectorControllerCfg;

static FluxVectorControllerCfg flux_vector_controller_cfg(double i_s_max);

/* 2DOF PI speed controller (SpeedController); alpha_i = NAN means alpha_s */
static PIController speed_controller(double J, double alpha_s, double alpha_i,
                                     double tau_M_max);

static void vector_control_system_init(VectorControlSystem *self,
                                       SynchronousMachinePars par,
                                       const FluxVectorControllerCfg *cfg,
                                       PIController speed_ctrl);
static void vector_control_system_compute_output(VectorControlSystem *self,
                                                 const Measurements *meas,
                                                 double w_M_ref);
static void vector_control_system_update(VectorControlSystem *self);

#endif
