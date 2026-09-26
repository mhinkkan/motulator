/*
 * Sensorless flux-vector control of synchronous machine drives, ported from
 * motulator. See sm_flux_vector.h.
 */

#include "sm_flux_vector.h"

/* Configuration ------------------------------------------------------------ */

static FluxVectorControllerCfg flux_vector_controller_cfg(double i_s_max)
{
    FluxVectorControllerCfg cfg;
    cfg.i_s_max = i_s_max;
    cfg.alpha_tau = 2.0 * M_PI * 100.0;
    cfg.alpha_psi = NAN;
    cfg.alpha_i = NAN;
    cfg.alpha_o = NAN;
    cfg.k_o[0] = NAN;
    cfg.k_o[1] = NAN;
    cfg.psi_s_min = NAN;
    cfg.psi_s_max = INFINITY;
    cfg.k_u = 0.9;
    cfg.k_mtpv = 0.85;
    cfg.J = NAN;
    cfg.T_s = 125e-6;
    return cfg;
}

static PIController speed_controller(double J, double alpha_s, double alpha_i,
                                     double tau_M_max)
{
    PIController ctrl;
    alpha_i = isnan(alpha_i) ? alpha_s : alpha_i;
    double k_p = (alpha_s + alpha_i) * J;
    double k_i = alpha_s * alpha_i * J;
    double k_t = alpha_s * J;
    pi_init(&ctrl, k_p, k_t, k_i / k_t, tau_M_max);
    return ctrl;
}

/* FluxObserver ------------------------------------------------------------- */

static void flux_observer_init(FluxObserver *self, SynchronousMachinePars par,
                               double k_theta, const double k_o[2])
{
    self->par = par;
    self->k_theta = k_theta;
    self->k_o[0] = k_o[0];
    self->k_o[1] = k_o[1];
    self->theta_m = 0.0;
    self->psi_s = par.psi_f;
}

static double complex flux_observer_aux_flux(const FluxObserver *self,
                                             double complex i_s,
                                             double complex psi_s)
{
    double L_dd = self->par.L_d, L_qq = self->par.L_q, L_dq = 0.0;
    return psi_s - L_qq * creal(i_s) - I * L_dd * cimag(i_s) + I * L_dq * conj(i_s);
}

static void flux_observer_compute_output(const FluxObserver *self,
                                         double complex u_s_ab, double complex i_s_ab,
                                         double w_M, ObserverOutputs *out)
{
    const SynchronousMachinePars *par = &self->par;
    double h = 0.0; /* Sensorless mode: model-based error signal only */
    out->psi_s = self->psi_s;
    out->psi_f = par->psi_f;
    out->h = h;

    /* Rotor speed */
    out->w_M = w_M;
    out->w_m = par->n_p * w_M;

    /* Coordinate system angle equals the estimated rotor angle */
    out->theta_c = out->theta_m = self->theta_m;

    /* Current and voltage vectors in (estimated) rotor coordinates */
    out->i_s = cexp(-I * out->theta_c) * i_s_ab;
    out->u_s = cexp(-I * out->theta_c) * u_s_ab;

    /* Flux estimation error */
    double complex psi_s_model = psi_s_dq(par, out->i_s);
    out->e_o = psi_s_model - out->psi_s;

    /* Auxiliary flux */
    out->psi_a = flux_observer_aux_flux(self, out->i_s, psi_s_model);

    /* Error signals for the rotor angle and PM-flux estimation */
    double complex ratio = (out->psi_a != 0.0) ? out->e_o / out->psi_a : 0.0;
    out->eps = -(1.0 - h) * cimag(ratio) / par->n_p;
    out->eps_f = -(1.0 - h) * creal(ratio);

    /* Angular speed of the coordinate system */
    out->w_c = out->w_m + self->k_theta * par->n_p * out->eps;

    /* Torque estimate */
    out->tau_M = 1.5 * par->n_p * cimag(out->i_s * conj(out->psi_s));
}

static void flux_observer_update(FluxObserver *self, double T_s,
                                 const ObserverOutputs *out)
{
    const SynchronousMachinePars *par = &self->par;

    double k_o1 = self->k_o[0] + self->k_o[1] * fabs(out->w_m);
    double complex k_o2;
    if (out->psi_a != 0.0) {
        k_o2 = (1.0 - out->h) * (k_o1 * out->psi_a / conj(out->psi_a));
    } else {
        k_o2 = (1.0 - out->h) * k_o1;
    }

    /* Update the state estimates (the PM-flux estimation gain k_f is zero) */
    double complex v = out->u_s - par->R_s * out->i_s - I * out->w_c * out->psi_s;
    self->psi_s += T_s * (v + k_o1 * out->e_o + k_o2 * conj(out->e_o));
    self->theta_m = wrap(self->theta_m + T_s * out->w_c);
}

/* SpeedFluxObserver -------------------------------------------------------- */

static void speed_flux_observer_init(SpeedFluxObserver *self,
                                     SynchronousMachinePars par, double alpha_o,
                                     const double k_o[2], double J)
{
    /* Observer gains for critically damped dynamics */
    double k_theta, k_w, k_tau;
    if (J <= 0.0) {
        k_theta = 2.0 * alpha_o;
        k_w = pow(alpha_o, 2.0);
        k_tau = 0.0;
    } else {
        k_theta = 3.0 * alpha_o;
        k_w = 3.0 * pow(alpha_o, 2.0);
        k_tau = J * pow(alpha_o, 3.0);
    }
    speed_observer_init(&self->speed_observer, k_w, k_tau, J);
    flux_observer_init(&self->flux_observer, par, k_theta, k_o);
}

static void speed_flux_observer_compute_output(const SpeedFluxObserver *self,
                                               double complex u_s_ab,
                                               double complex i_s_ab,
                                               ObserverOutputs *out)
{
    double w_M = self->speed_observer.w_M;
    double tau_L = self->speed_observer.tau_L;
    flux_observer_compute_output(&self->flux_observer, u_s_ab, i_s_ab, w_M, out);
    out->tau_L = tau_L;
}

static void speed_flux_observer_update(SpeedFluxObserver *self, double T_s,
                                       const ObserverOutputs *out)
{
    speed_observer_update(&self->speed_observer, T_s, out->eps, out->tau_M);
    flux_observer_update(&self->flux_observer, T_s, out);
}

/* ReferenceGenerator ------------------------------------------------------- */

static void reference_gen_init(ReferenceGenerator *self,
                               const SynchronousMachinePars *par, double i_s_max,
                               double psi_s_min, double psi_s_max, double k_u,
                               double k_mtpv)
{
    self->k_u = k_u;
    self->k_mtpv = k_mtpv;
    self->psi_s_min = isnan(psi_s_min) ? par->psi_f : psi_s_min;
    self->psi_s_max = psi_s_max;
    ReferenceLUTs luts;
    compute_reference_luts(par, i_s_max, &luts);
    self->psi_s_mtpa = luts.psi_s_mtpa;
    self->tau_M_cl = luts.tau_M_cl;
    self->tau_M_mtpv = luts.tau_M_mtpv;
}

static double reference_gen_max_flux(const ReferenceGenerator *self, double w_m,
                                     double u_dc)
{
    double u_s_max = self->k_u * u_dc / sqrt(3.0);
    return (w_m != 0.0) ? u_s_max / fabs(w_m) : INFINITY;
}

static void reference_gen_compute_flux_and_torque_refs(const ReferenceGenerator *self,
                                                       double tau_M_ref, double w_m,
                                                       double u_dc, double *psi_s_ref,
                                                       double *tau_M_ref_lim)
{
    /* MTPA flux */
    double psi_s_abs_ref = clip(lut_interp(&self->psi_s_mtpa, fabs(tau_M_ref)),
                                self->psi_s_min, self->psi_s_max);

    /* Maximum flux (field weakening) */
    psi_s_abs_ref = fmin(psi_s_abs_ref, reference_gen_max_flux(self, w_m, u_dc));

    /* Current limit */
    double tau_M_cl = lut_interp(&self->tau_M_cl, psi_s_abs_ref);
    tau_M_ref = fmin(tau_M_cl, fabs(tau_M_ref)) * sign(tau_M_ref);

    /* MTPV limit */
    double tau_M_mtpv = lut_interp(&self->tau_M_mtpv, psi_s_abs_ref);
    if (tau_M_mtpv > 0.0) {
        tau_M_ref = fmin(self->k_mtpv * tau_M_mtpv, fabs(tau_M_ref)) * sign(tau_M_ref);
    }

    *psi_s_ref = psi_s_abs_ref;
    *tau_M_ref_lim = tau_M_ref;
}

/* FluxTorqueController ----------------------------------------------------- */

static void flux_torque_ctrl_init(FluxTorqueController *self,
                                  SynchronousMachinePars par, double alpha_psi,
                                  double alpha_tau, double alpha_i)
{
    self->par = par;
    self->alpha_psi = alpha_psi;
    self->alpha_tau = alpha_tau;
    self->alpha_i = alpha_i;
    self->x_psi = alpha_i * par.psi_f;
    self->x_tau = 0.0;
    self->i_a = 0.0;
    self->v = 0.0;
}

static double complex flux_torque_ctrl_aux_current(const FluxTorqueController *self,
                                                   double complex psi_s,
                                                   double complex i_s)
{
    double L_dd = self->par.L_d, L_qq = self->par.L_q, L_dq = 0.0;
    double det_L = L_dd * L_qq - L_dq * L_dq;
    return (L_dd * creal(psi_s) + I * L_qq * cimag(psi_s) + I * L_dq * conj(psi_s))
               / det_L
           - i_s;
}

static double complex flux_torque_ctrl_compute_output(FluxTorqueController *self,
                                                      double psi_s_ref,
                                                      double tau_M_ref,
                                                      const ObserverOutputs *fbk)
{
    const SynchronousMachinePars *par = &self->par;
    double psi_s_abs = cabs(fbk->psi_s);

    /* Auxiliary current and torque-production factor */
    double complex i_a = flux_torque_ctrl_aux_current(self, fbk->psi_s, fbk->i_s);
    double c_tau = 1.5 * par->n_p * creal(i_a * conj(fbk->psi_s));

    /* Directions */
    double complex t_psi = (c_tau > 0.0) ? 1.5 * par->n_p * psi_s_abs * i_a / c_tau : 1.0;
    double complex t_tau = (c_tau > 0.0) ? I * fbk->psi_s / c_tau : 0.0;

    /* Error signals */
    double e_psi = psi_s_ref - psi_s_abs;
    double e_tau = tau_M_ref - fbk->tau_M;
    double complex e_u = self->alpha_psi * e_psi * t_psi + self->alpha_tau * e_tau * t_tau;
    double complex u_i = self->x_psi * t_psi + self->x_tau * t_tau;
    double complex e_v = u_i - self->alpha_i * (psi_s_abs * t_psi + fbk->tau_M * t_tau);

    /* Voltage reference */
    double complex v = par->R_s * fbk->i_s + I * fbk->w_m * fbk->psi_s + e_v;
    double complex u_s_ref = v + e_u;

    /* Workspace variables for the update function */
    self->i_a = i_a;
    self->v = v;

    return u_s_ref;
}

static void flux_torque_ctrl_update(FluxTorqueController *self, double T_s,
                                    const ObserverOutputs *fbk)
{
    const SynchronousMachinePars *par = &self->par;
    double psi_s_abs = cabs(fbk->psi_s);
    /* Error signal and gains */
    double complex e = fbk->u_s - self->v;
    double k_psi = (psi_s_abs > 0.0) ? self->alpha_i / psi_s_abs : 0.0;
    double k_tau = 1.5 * par->n_p * self->alpha_i;
    /* Update the integral states */
    self->x_psi += T_s * k_psi * creal(fbk->psi_s * conj(e));
    self->x_tau += T_s * k_tau * creal(I * self->i_a * conj(e));
}

/* FluxVectorController ----------------------------------------------------- */

static void flux_vector_ctrl_init(FluxVectorController *self,
                                  SynchronousMachinePars par,
                                  const FluxVectorControllerCfg *cfg)
{
    /* Resolve the defaults, as in FluxVectorControllerCfg.__post_init__ and
     * FluxVectorController.__init__ */
    double alpha_psi = isnan(cfg->alpha_psi) ? cfg->alpha_tau : cfg->alpha_psi;
    double alpha_i = isnan(cfg->alpha_i) ? cfg->alpha_tau : cfg->alpha_i;
    double J = isnan(cfg->J) ? 0.0 : cfg->J; /* Zero means None */
    double alpha_o = cfg->alpha_o;
    if (isnan(alpha_o)) {
        double alpha = 2.0 * M_PI * 50.0; /* Sensorless mode */
        alpha_o = (J > 0.0) ? alpha / 3.0 : alpha;
    }
    /* Default observer gain of create_speed_flux_observer (sensorless mode) */
    double k_o[2] = {cfg->k_o[0], cfg->k_o[1]};
    if (isnan(k_o[0]) || isnan(k_o[1])) {
        k_o[0] = 0.25 * par.R_s * (1.0 / par.L_d + 1.0 / par.L_q);
        k_o[1] = 0.2;
    }

    self->T_s = cfg->T_s;
    reference_gen_init(&self->reference_gen, &par, cfg->i_s_max, cfg->psi_s_min,
                       cfg->psi_s_max, cfg->k_u, cfg->k_mtpv);
    flux_torque_ctrl_init(&self->flux_torque_ctrl, par, alpha_psi, cfg->alpha_tau,
                          alpha_i);
    speed_flux_observer_init(&self->observer, par, alpha_o, k_o, J);
}

static void flux_vector_ctrl_compute_output(FluxVectorController *self,
                                            double tau_M_ref,
                                            const ObserverOutputs *fbk,
                                            References *ref)
{
    ref->T_s = self->T_s;
    reference_gen_compute_flux_and_torque_refs(&self->reference_gen, tau_M_ref,
                                               fbk->w_m, fbk->u_dc, &ref->psi_s,
                                               &ref->tau_M);
    ref->u_s = flux_torque_ctrl_compute_output(&self->flux_torque_ctrl, ref->psi_s,
                                               ref->tau_M, fbk);
}

static void flux_vector_ctrl_update(FluxVectorController *self, const References *ref,
                                    const ObserverOutputs *fbk)
{
    speed_flux_observer_update(&self->observer, ref->T_s, fbk);
    flux_torque_ctrl_update(&self->flux_torque_ctrl, ref->T_s, fbk);
}

/* VectorControlSystem ------------------------------------------------------ */

static void vector_control_system_init(VectorControlSystem *self,
                                       SynchronousMachinePars par,
                                       const FluxVectorControllerCfg *cfg,
                                       PIController speed_ctrl)
{
    pwm_init(&self->pwm, 1.5);
    flux_vector_ctrl_init(&self->vector_ctrl, par, cfg);
    self->speed_ctrl = speed_ctrl;
    self->speed_ctrl.u_i = 0.0;
    self->speed_ctrl.v = 0.0;
}

static void vector_control_system_compute_output(VectorControlSystem *self,
                                                 const Measurements *meas,
                                                 double w_M_ref)
{
    ObserverOutputs *fbk = &self->fbk;
    References *ref = &self->ref;

    /* Feedback signals */
    double complex u_c_ab = self->pwm.realized_voltage;
    speed_flux_observer_compute_output(&self->vector_ctrl.observer, u_c_ab,
                                       meas->i_c_ab, fbk);
    fbk->u_dc = meas->u_dc;

    /* Speed controller and vector controller */
    double tau_M_ref = pi_compute_output(&self->speed_ctrl, w_M_ref, fbk->w_M, 0.0);
    flux_vector_ctrl_compute_output(&self->vector_ctrl, tau_M_ref, fbk, ref);

    /* Duty ratios for the PWM */
    double complex u_s_ab_ref = cexp(I * fbk->theta_c) * ref->u_s;
    ref->u_c_ab = pwm_compute_output(&self->pwm, ref->T_s, u_s_ab_ref, fbk->u_dc,
                                     fbk->w_c, ref->d_abc);
    ref->w_M = w_M_ref;
}

static void vector_control_system_update(VectorControlSystem *self)
{
    pwm_update(&self->pwm, self->ref.u_c_ab);
    flux_vector_ctrl_update(&self->vector_ctrl, &self->ref, &self->fbk);
    pi_update(&self->speed_ctrl, self->ref.T_s, self->ref.tau_M);
}
