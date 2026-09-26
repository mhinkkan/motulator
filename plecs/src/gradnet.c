/*
 * Gradient networks (GradNets) for magnetics modeling, ported from motulator. See
 * gradnet.h.
 */

#include "gradnet.h"

/* GradNet.forward for a per-unit input x */
static void gradnet_forward(const GradNet *net, const double *x, double *z)
{
    int n = net->embed_dim, m = net->in_dim;
    double h[GRADNET_MAX_EMBED_DIM];
    double beta = exp(net->beta_log);

    /* Linear term (for the first mu_dim inputs) and bias */
    for (int k = 0; k < m; k++) {
        double mu = (k < net->mu_dim) ? exp(net->mu_log[k]) : 0.0;
        z[k] = mu * x[k] + net->bias[k];
    }

    /* Module: W^T act(W x + b) */
    for (int j = 0; j < n; j++) {
        double w_x = net->b[j];
        for (int k = 0; k < m; k++) {
            w_x += net->W[j][k] * x[k];
        }
        h[j] = beta * w_x;
    }
    if (net->activation == GRADNET_PNORM_GRADIENT) {
        /* PNormGradient: x^q/(1 + sum(x^(q + 1)))^(q/(q + 1)) with q = p - 1 */
        int q = net->p - 1;
        double norm = 1.0;
        for (int j = 0; j < n; j++) {
            norm += pow(h[j], q + 1);
        }
        norm = pow(norm, (double)q / (q + 1));
        for (int j = 0; j < n; j++) {
            h[j] = pow(h[j], q) / norm;
        }
    } else {
        /* Softmax */
        double h_max = h[0], sum = 0.0;
        for (int j = 1; j < n; j++) {
            h_max = fmax(h_max, h[j]);
        }
        for (int j = 0; j < n; j++) {
            h[j] = exp(h[j] - h_max);
            sum += h[j];
        }
        for (int j = 0; j < n; j++) {
            h[j] /= sum;
        }
    }
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            z[k] += net->W[j][k] * h[j];
        }
    }
}

static double complex gradnet_map(const GradNet *net, double complex x)
{
    /* Evaluate at the input and its conjugate, and symmetrize about the d-axis */
    double x1[2] = {creal(x) / net->in_base, cimag(x) / net->in_base};
    double x2[2] = {x1[0], -x1[1]};
    double z1[2], z2[2];
    gradnet_forward(net, x1, z1);
    gradnet_forward(net, x2, z2);
    double complex y = 0.5 * ((z1[0] + I * z1[1]) + conj(z2[0] + I * z2[1]));
    return y * net->out_base;
}

static void gradnet_current_map_harmonics(const GradNet *net, int k,
                                          double complex psi_s, double theta_m,
                                          double complex *i_s, double *tau_m)
{
    /* Evaluate at the input and its conjugate, and symmetrize about the d-axis */
    double complex psi = psi_s / net->in_base;
    double complex exp_j_k_theta = cexp(I * k * theta_m);
    double x1[4] = {creal(psi), cimag(psi), creal(exp_j_k_theta),
                    cimag(exp_j_k_theta)};
    double x2[4] = {x1[0], -x1[1], x1[2], -x1[3]};
    double z1[4], z2[4];
    gradnet_forward(net, x1, z1);
    gradnet_forward(net, x2, z2);
    double complex i = 0.5 * ((z1[0] + I * z1[1]) + conj(z2[0] + I * z2[1]));
    double dW_dcos = 0.5 * (z1[2] + z2[2]);
    double dW_dsin = 0.5 * (z1[3] - z2[3]);

    /* Torque in per-unit values, including the reluctance torque due to the spatial
     * harmonics */
    double dW_dtheta =
        k * (creal(exp_j_k_theta) * dW_dsin - cimag(exp_j_k_theta) * dW_dcos);
    double tau = cimag(i * conj(psi)) - dW_dtheta;

    /* Scale back to physical units */
    *i_s = i * net->out_base;
    *tau_m = tau * 1.5 * net->in_base * net->out_base;
}
