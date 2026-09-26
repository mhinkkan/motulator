/*
 * Gradient networks (GradNets) for magnetics modeling, ported from motulator.
 *
 * Python counterpart:
 *   motulator/drive/gradnet/_gn.py  GradNet, FluxMap, CurrentMap,
 *                                   CurrentMapWithHarmonics
 *
 * Only the inference is ported (the networks are trained in motulator). The
 * computations use double precision, while PyTorch uses single precision.
 */

#ifndef MOTULATOR_GRADNET_H
#define MOTULATOR_GRADNET_H

#include <complex.h>
#include <math.h>

#define GRADNET_MAX_EMBED_DIM 64
#define GRADNET_MAX_IN_DIM 4

typedef enum {
    GRADNET_PNORM_GRADIENT = 1,
    GRADNET_SOFTMAX = 2
} GradNetActivation;

/* GradNet with a single module. The input dimension is 2 for the flux and current
 * maps and 4 for the current map with spatial harmonics. The linear term is applied
 * to the first mu_dim inputs. */
typedef struct {
    int in_dim;
    int mu_dim;
    int embed_dim;
    double W[GRADNET_MAX_EMBED_DIM][GRADNET_MAX_IN_DIM];
    double b[GRADNET_MAX_EMBED_DIM];
    double mu_log[GRADNET_MAX_IN_DIM];
    double bias[GRADNET_MAX_IN_DIM];
    GradNetActivation activation;
    double beta_log;
    int p; /* Exponent of the p-norm gradient activation */
    double in_base;  /* Base value of the input */
    double out_base; /* Base value of the output */
} GradNet;

/* Evaluate the map (FluxMap or CurrentMap in motulator), symmetrized about the
 * d-axis. The input and output are in SI units. */
static double complex gradnet_map(const GradNet *net, double complex x);

/* Evaluate the current map with spatial harmonics (CurrentMapWithHarmonics in
 * motulator) of order k. Returns the stator current (A) in rotor coordinates and
 * the electromagnetic torque (Nm) per pole pair. */
static void gradnet_current_map_harmonics(const GradNet *net, int k,
                                          double complex psi_s, double theta_m,
                                          double complex *i_s, double *tau_m);

#endif
