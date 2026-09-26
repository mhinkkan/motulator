/*
 * Common control functions and classes, ported from motulator.
 *
 * Python counterparts:
 *   motulator/common/utils/_utils.py        abc2complex, complex2abc, wrap, clip, sign
 *   motulator/common/control/_controllers.py PIController
 *   motulator/common/control/_pwm.py         PWM (MPE overmodulation)
 *   motulator/drive/control/_common.py       SpeedController, SpeedObserver
 *
 * Complex space vectors use peak-value scaling, as in motulator.
 */

#ifndef MOTULATOR_COMMON_H
#define MOTULATOR_COMMON_H

#include <complex.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Utility functions */
static double complex abc2complex(const double u[3]);
static void complex2abc(double complex u, double u_abc[3]);
static double wrap(double theta);
static double clip(double value, double min_value, double max_value);
static double sign(double x);

/* Lookup table with linear interpolation, equivalent to numpy.interp */
#define LUT_MAX_SIZE 64

typedef struct {
    int n;
    double x[LUT_MAX_SIZE]; /* Increasing sample points */
    double y[LUT_MAX_SIZE];
} LUT;

static double lut_interp(const LUT *lut, double x);

/* Brent's root finding in a bracketing interval, as scipy.optimize.brentq with the
 * default tolerances (used via scipy.optimize.root_scalar in motulator) */
typedef double (*ScalarFunction)(double x, void *data);

static double brentq(ScalarFunction f, double xa, double xb, void *data);

/* 2DOF PI controller */
typedef struct {
    double k_p;
    double k_t;
    double alpha_i; /* Inverse of the integration time */
    double u_max;
    /* States and workspace variables */
    double u_i;
    double v;
} PIController;

static void pi_init(PIController *self, double k_p, double k_t, double alpha_i,
                    double u_max);
static double pi_compute_output(PIController *self, double y_ref, double y,
                                double u_ff);
static void pi_update(PIController *self, double T_s, double u);

/* Speed observer (mechanical system model is used if J > 0) */
typedef struct {
    double k_w;
    double k_tau;
    double J;
    /* States */
    double w_M;
    double tau_L;
} SpeedObserver;

static void speed_observer_init(SpeedObserver *self, double k_w, double k_tau,
                                double J);
static void speed_observer_update(SpeedObserver *self, double T_s, double eps,
                                  double tau_M);

/* Space-vector PWM with the minimum-phase-error (MPE) overmodulation */
typedef struct {
    double k_comp;
    /* States */
    double complex realized_voltage;
    double complex old_u_c_ab;
} PWM;

static void pwm_init(PWM *self, double k_comp);
static double complex pwm_compute_output(const PWM *self, double T_s,
                                         double complex u_c_ref_ab, double u_dc,
                                         double w, double d_abc[3]);
static void pwm_update(PWM *self, double complex u_c_ab);

#endif
