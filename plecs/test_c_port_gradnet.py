"""
Test the C port with a GradNet flux map against motulator.

This test uses the control system of the example
examples/drive/gradnet/plot_6kw_pmsyrm_gn_fvc_fem_harm.py: sensored flux-vector
control, where the estimated machine model is a GradNet flux map trained on FEM data.

The C port evaluates the GradNet in double precision, while motulator (PyTorch) uses
single precision. The finite-difference incremental inductances (with the step of
1e-3 A) amplify the single-precision rounding errors to about 1 %. Hence, the C port
is compared with motulator using the same GradNet evaluated in double precision
(`FluxMap64`), which should agree to rounding errors. The differences to the standard
single-precision evaluation are printed for information.

Run from the repository root (requires gcc and PyTorch):

    python plecs/test_c_port_gradnet.py

"""

# %%
import ctypes
import subprocess
import sys
import tempfile
from math import pi
from pathlib import Path
from typing import Any, cast

import numpy as np

import motulator.drive.control.sm as control
import motulator.drive.gradnet as gn
from motulator.common.utils import abc2complex
from motulator.common.utils._utils import wrap
from motulator.drive import utils
from motulator.drive.control._base import Measurements

sys.path.insert(0, str(Path(__file__).parent))
from gradnet64 import CurrentMapWithHarmonics64, FluxMap64  # noqa: E402
from motulator_plecs import export_gradnet  # noqa: E402

SRC_DIR = Path(__file__).parent / "src"
MODEL_DIR = (
    Path(__file__).parents[1] / "examples" / "drive" / "gradnet" / "trained_models"
)

CAPI = r"""
#include "common.c"
#include "gradnet.c"
#include "sm_parameters.c"
#include "sm_machine.c"
#include "sm_control_loci.c"
#include "sm_flux_vector.c"

static GradNet net;
static SynchronousMachinePars par;
static SpatialSaturatedSynchronousMachinePars plant_par;
static VectorControlSystem ctrl;

void set_net(int m, int mu_dim, int n, const double *W, const double *b,
             const double *mu_log, const double *bias, int activation,
             double beta_log, int p, double in_base, double out_base)
{
    net.in_dim = m;
    net.mu_dim = mu_dim;
    net.embed_dim = n;
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < m; k++) {
            net.W[j][k] = W[m * j + k];
        }
        net.b[j] = b[j];
    }
    for (int k = 0; k < m; k++) {
        net.mu_log[k] = (k < mu_dim) ? mu_log[k] : 0.0;
        net.bias[k] = bias[k];
    }
    net.activation = (GradNetActivation)activation;
    net.beta_log = beta_log;
    net.p = p;
    net.in_base = in_base;
    net.out_base = out_base;
}

void set_plant(double n_p, double R_s, int k, double *out)
{
    plant_par = spatial_saturated_synchronous_machine_pars(n_p, R_s, &net, k);
    out[0] = plant_par.psi_f;
}

void eval_plant(double psi_d, double psi_q, double theta_M, double w_M,
                double u_a, double u_b, double tau_L, double J, double *out)
{
    double complex i_s_dq, d_psi;
    double tau_M, d_theta_M, d_w_M;
    double complex psi = psi_d + I * psi_q;
    machine_magnetic_map(&plant_par, psi, plant_par.n_p * theta_M, &i_s_dq, &tau_M);
    machine_rhs(&plant_par, J, u_a + I * u_b, tau_L, psi, theta_M, w_M, &d_psi,
                &d_theta_M, &d_w_M);
    double o[7] = {creal(i_s_dq), cimag(i_s_dq), tau_M, creal(d_psi), cimag(d_psi),
                   d_theta_M, d_w_M};
    for (int k = 0; k < 7; k++) {
        out[k] = o[k];
    }
}

void set_par(double n_p, double R_s, double *out)
{
    par = saturated_synchronous_machine_pars(n_p, R_s, &net);
    out[0] = par.psi_f;
    out[1] = par.L_d0;
    out[2] = par.L_q0;
}

void eval(double i_d, double i_q, double *out)
{
    double complex i_s = i_d + I * i_q;
    double complex psi_s = psi_s_dq(&par, i_s);
    IncrIndMat L = incr_ind_mat(&par, i_s);
    double complex i_s_it = iterate_i_s_dq(&par, psi_s);
    out[0] = creal(psi_s);
    out[1] = cimag(psi_s);
    out[2] = L.L_dd;
    out[3] = L.L_dq;
    out[4] = L.L_qq;
    out[5] = creal(i_s_it);
    out[6] = cimag(i_s_it);
}

void luts(double i_s_max, double *out)
{
    ReferenceLUTs luts;
    compute_reference_luts(&par, i_s_max, &luts);
    const LUT *all[3] = {&luts.psi_s_mtpa, &luts.tau_M_cl, &luts.tau_M_mtpv};
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < NUM_LOCUS; k++) {
            out[2 * NUM_LOCUS * i + k] = all[i]->x[k];
            out[2 * NUM_LOCUS * i + NUM_LOCUS + k] = all[i]->y[k];
        }
    }
}

void init(const double *c, const double *s)
{
    FluxVectorControllerCfg cfg = flux_vector_controller_cfg(c[0]);
    cfg.alpha_i = c[1];
    cfg.alpha_o = c[2];
    cfg.J = c[3];
    cfg.sensorless = (int)c[4];
    PIController speed_ctrl = speed_controller(s[0], s[1], NAN, INFINITY);
    vector_control_system_init(&ctrl, par, &cfg, speed_ctrl);
}

void step(const double *i_s_abc, double u_dc, double w_M_ref, double theta_M,
          double *out)
{
    Measurements meas = {abc2complex(i_s_abc), u_dc, theta_M};
    vector_control_system_compute_output(&ctrl, &meas, w_M_ref);
    out[0] = ctrl.ref.d_abc[0];
    out[1] = ctrl.ref.d_abc[1];
    out[2] = ctrl.ref.d_abc[2];
    out[3] = ctrl.fbk.w_M;
    out[4] = ctrl.fbk.tau_M;
    out[5] = ctrl.ref.tau_M;
    out[6] = ctrl.ref.psi_s;
    out[7] = ctrl.fbk.theta_m;
    vector_control_system_update(&ctrl);
}
"""


def compile_library() -> ctypes.CDLL:
    """Compile the C sources into a shared library."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "capi.c").write_text(CAPI)
    lib = tmp / "libmotulator.so"
    cmd = ["gcc", "-std=c99", "-O2", "-shared", "-fPIC", "-Wall"]
    cmd += ["-Wno-unused-function", f"-I{SRC_DIR}", str(tmp / "capi.c")]
    subprocess.run([*cmd, "-lm", "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib))


def _arr(values: Any) -> Any:
    values = np.asarray(values, dtype=float).ravel()
    return (ctypes.c_double * len(values))(*values)


def _d(x: float) -> ctypes.c_double:
    return ctypes.c_double(x)


def set_net(dll: ctypes.CDLL, g: dict[str, Any]) -> None:
    """Set the GradNet of the C port."""
    dll.set_net(
        g["in_dim"],
        g["mu_dim"],
        len(g["b"]),
        _arr(g["W"]),
        _arr(g["b"]),
        _arr(g["mu_log"]),
        _arr(g["bias"]),
        g["activation"],
        _d(g["beta_log"]),
        g["p"],
        _d(g["in_base"]),
        _d(g["out_base"]),
    )


# %%
# Machine model and control system of the example
nom = utils.NominalValues(U=460, I=8.8, f=60, P=5.6e3, tau=29.7)
base = utils.BaseValues.from_nominal(nom, n_p=2)
FLUX_MAP = MODEL_DIR / "baldor_fem_flux_map_pnorm_d12_sub20.pth"
CURRENT_MAP = MODEL_DIR / "baldor_fem_curr_map_harm_softmax_d48_sub10.pth"


def setup(double: bool = True) -> tuple[ctypes.CDLL, Any]:
    """Load the flux map in motulator and in the C port."""
    model = gn.load_gradnet(FLUX_MAP, activation=gn.PNormGradient)
    flux_map = FluxMap64(model) if double else gn.FluxMap(model)
    est_par = control.SaturatedSynchronousMachinePars(
        n_p=2, R_s=0.63, psi_s_dq_fcn=flux_map
    )
    dll = compile_library()
    set_net(dll, export_gradnet(flux_map))
    out = (ctypes.c_double * 3)()
    dll.set_par(_d(2), _d(0.63), out)
    print("  psi_f, L_d0, L_q0: C", np.array(out), "Python", end=" ")
    print(np.array([est_par.psi_f, est_par.L_d0, est_par.L_q0]))
    return dll, est_par


def test_machine_model() -> None:
    """Flux linkage, incremental inductances, and iterated current."""
    dll, est_par = setup()
    _, est_par32 = setup(double=False)
    out = (ctypes.c_double * 7)()
    err_psi, err_L, err_i, err_L32 = 0.0, 0.0, 0.0, 0.0
    for i_d in np.linspace(-20, 20, 9):
        for i_q in np.linspace(-25, 25, 11):
            dll.eval(_d(i_d), _d(i_q), out)
            i_s = i_d + 1j * i_q
            psi_py = complex(est_par.psi_s_dq(i_s))
            L_py = est_par.incr_ind_mat(i_s)
            err_psi = max(err_psi, abs(out[0] + 1j * out[1] - psi_py) / abs(psi_py))
            L_c = np.array([[out[2], out[3]], [out[3], out[4]]])
            err_L = max(err_L, np.max(np.abs(L_c - L_py)) / np.max(np.abs(L_py)))
            err_i = max(err_i, abs(out[5] + 1j * out[6] - i_s))
            L_32 = est_par32.incr_ind_mat(i_s)
            err_L32 = max(err_L32, np.max(np.abs(L_c - L_32)) / np.max(np.abs(L_32)))
    print(f"  psi_s_dq: max relative error {err_psi:.3g}")
    print(f"  incr_ind_mat: max relative error {err_L:.3g}")
    print(f"  iterate_i_s_dq: max error {err_i:.3g} A")
    print(f"  (incr_ind_mat vs. single-precision motulator: {err_L32:.3g})")
    assert err_psi < 1e-12
    assert err_L < 1e-8
    assert err_i < 1e-9


def test_luts() -> None:
    """Lookup tables of the reference generator."""
    dll, est_par = setup()
    cfg = control.FluxVectorControllerCfg(i_s_max=2 * base.i, sensorless=False)
    rg = cast(Any, control.FluxVectorController(est_par, cfg).reference_gen)
    out = (ctypes.c_double * (6 * 16))()
    dll.luts(_d(2 * base.i), out)
    luts = np.array(out).reshape(3, 2, 16)
    for i, (f, name) in enumerate(
        [(rg.psi_s_mtpa, "psi_s_mtpa"), (rg.tau_M_cl, "tau_M_cl"), (rg.tau_M_mtpv, "")]
    ):
        err = np.max(np.abs(f(luts[i, 0]) - luts[i, 1])) / np.max(np.abs(luts[i, 1]))
        print(f"  {name or 'tau_M_mtpv'}: max relative error {err:.3g}")
        assert err < 1e-8


def test_control_system() -> None:
    """Sensored control system of the example for a sequence of measurements."""
    dll, est_par = setup()
    cfg = control.FluxVectorControllerCfg(
        i_s_max=2 * base.i, alpha_i=0, alpha_o=2 * pi * 8, J=0.05, sensorless=False
    )
    speed = {"J": 0.05, "alpha_s": 2 * pi * 4}
    ctrl = control.VectorControlSystem(
        control.FluxVectorController(est_par, cfg), control.SpeedController(**speed)
    )
    dll.init(
        _arr([cfg.i_s_max, 0.0, 2 * pi * 8, 0.05, 0.0]), _arr(list(speed.values()))
    )

    # Measurements: accelerating rotor, current vector in rotor coordinates
    T_s, n = cfg.T_s, 3000
    t = np.arange(n) * T_s
    w_M = 100.0 * t / t[-1]
    theta_M = np.cumsum(w_M) * T_s
    i_s_dq = (5 + 10 * np.sin(2 * pi * 5 * t)) + 1j * (8 + 12 * np.cos(2 * pi * 3 * t))
    i_s_ab = np.exp(1j * 2 * theta_M) * i_s_dq
    ctrl.set_speed_ref(lambda t_: 50.0 if t_ > 0.05 else 0.0)

    out = (ctypes.c_double * 8)()
    res_c, res_py = np.zeros((n, 8)), np.zeros((n, 8))
    for k in range(n):
        i_abc = [
            i_s_ab[k].real,
            0.5 * (-i_s_ab[k].real + np.sqrt(3) * i_s_ab[k].imag),
            0.5 * (-i_s_ab[k].real - np.sqrt(3) * i_s_ab[k].imag),
        ]
        th = float(wrap(theta_M[k]))
        meas = Measurements(abc2complex(i_abc), 540.0, w_M[k], th)
        fbk = cast(Any, ctrl.get_feedback(meas))
        ref = cast(Any, ctrl.compute_output(fbk))
        ctrl.update(ref, fbk)
        res_py[k] = [*ref.d_abc, fbk.w_M, fbk.tau_M, ref.tau_M, ref.psi_s, fbk.theta_m]
        # Same speed reference as in motulator (evaluated at the controller time)
        dll.step(_arr(i_abc), _d(540.0), _d(ref.w_M), _d(th), out)
        res_c[k] = np.array(out)
    names = ["d_a", "d_b", "d_c", "w_M", "tau_M", "tau_M_ref", "psi_s_ref", "theta_m"]
    dev = np.max(np.abs(res_c - res_py) / np.maximum(np.abs(res_py), 1.0), axis=1)
    k0 = int(np.argmax(dev > 1e-3)) if np.any(dev > 1e-3) else -1
    if k0 >= 0:
        print(f"  first deviation at k = {k0}, t = {t[k0]:.5f}")
        for kk in range(max(k0 - 2, 0), k0 + 2):
            print("   C ", kk, np.round(res_c[kk], 5))
            print("   Py", kk, np.round(res_py[kk], 5))
    err = np.max(np.abs(res_c - res_py), axis=0)
    scale = np.max(np.abs(res_py), axis=0)
    for name, e, s in zip(names, err, scale, strict=True):
        print(f"  {name}: max error {e:.3g} (max value {s:.3g})")
    assert np.all(err <= 1e-8 * np.maximum(scale, 1.0))


def test_plant() -> None:
    """Machine model with the GradNet current map with spatial harmonics."""
    from motulator.drive import model as mdl_model  # noqa: PLC0415

    current_map = CurrentMapWithHarmonics64(
        gn.load_gradnet(CURRENT_MAP, activation=gn.Softmax)
    )
    par = mdl_model.SpatialSaturatedSynchronousMachinePars(
        n_p=2, R_s=0.63, magnetic_map_fcn=current_map
    )
    machine = mdl_model.SynchronousMachine(par)
    dll = compile_library()
    set_net(dll, export_gradnet(current_map))
    out = (ctypes.c_double * 7)()
    dll.set_plant(_d(2), _d(0.63), 6, out)
    print(f"  psi_f: C {out[0]:.12f}, Python {par.psi_f:.12f}")
    assert abs(out[0] - par.psi_f) < 1e-9

    rng = np.random.default_rng(1)
    err = np.zeros(4)
    for _ in range(200):
        psi = rng.uniform(-0.3, 1.0) + 1j * rng.uniform(-0.8, 0.8)
        theta_M, w_M = rng.uniform(-pi, pi), rng.uniform(-200, 200)
        u_s_ab = complex(rng.uniform(-300, 300), rng.uniform(-300, 300))
        dll.eval_plant(
            _d(psi.real),
            _d(psi.imag),
            _d(theta_M),
            _d(w_M),
            _d(u_s_ab.real),
            _d(u_s_ab.imag),
            _d(0.0),
            _d(0.05),
            out,
        )
        # motulator
        machine.state.psi_s_dq = psi
        machine.state.exp_j_theta_m = np.exp(1j * 2 * theta_M)
        machine.inp.u_s_ab, machine.inp.w_M = u_s_ab, w_M
        machine.set_outputs(0.0)
        d_psi = machine.rhs(0.0)[0]
        err = np.maximum(
            err,
            [
                abs(out[0] + 1j * out[1] - machine.out.i_s_dq),
                abs(out[2] - machine.out.tau_M),
                abs(out[3] + 1j * out[4] - d_psi),
                abs(out[6] - (machine.out.tau_M - 0.0) / 0.05),
            ],
        )
    for name, e in zip(["i_s_dq", "tau_M", "d_psi_s_dq", "d_w_M"], err, strict=True):
        print(f"  {name}: max error {e:.3g}")
    assert np.all(err < 1e-9)


# %%
if __name__ == "__main__":
    for test in (test_machine_model, test_luts, test_control_system, test_plant):
        print(f"{test.__name__}:")
        test()
        print("  passed")
