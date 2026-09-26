"""
Test the C port of the motulator control algorithms against motulator.

The C sources in `src` are compiled into a shared library (requires gcc), and the
results are compared with motulator: the root finding, the lookup tables of the
reference generator, and the outputs of the complete control system for a sequence
of measurements. PLECS is not needed.

Run from the repository root:

    python plecs/test_c_port.py

"""

# %%
import ctypes
import subprocess
import tempfile
from math import inf, pi
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq

import motulator.drive.control.sm as control
from motulator.common.utils import abc2complex
from motulator.drive import model
from motulator.drive.control._base import Measurements

SRC_DIR = Path(__file__).parent / "src"

CAPI = r"""
#include "common.c"
#include "gradnet.c"
#include "sm_parameters.c"
#include "sm_control_loci.c"
#include "sm_flux_vector.c"

static VectorControlSystem ctrl;

static double test_function(double x, void *data)
{
    double c = *(double *)data;
    return cos(x) - c * x;
}

double test_brentq(double a, double b, double c)
{
    return brentq(test_function, a, b, &c);
}

void test_luts(const double *p, double i_s_max, double *out)
{
    SynchronousMachinePars par = {p[0], p[1], p[2], p[3], p[4]};
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

void init(const double *p, const double *c, const double *s)
{
    SynchronousMachinePars par = {p[0], p[1], p[2], p[3], p[4]};
    FluxVectorControllerCfg cfg = flux_vector_controller_cfg(c[0]);
    cfg.alpha_tau = c[1];
    cfg.alpha_psi = c[2];
    cfg.alpha_i = c[3];
    cfg.alpha_o = c[4];
    cfg.k_o[0] = c[5];
    cfg.k_o[1] = c[6];
    cfg.psi_s_min = c[7];
    cfg.psi_s_max = c[8];
    cfg.k_u = c[9];
    cfg.k_mtpv = c[10];
    cfg.J = c[11];
    cfg.T_s = c[12];
    PIController speed_ctrl = speed_controller(s[0], s[1], s[2], s[3]);
    vector_control_system_init(&ctrl, par, &cfg, speed_ctrl);
}

void step(const double *i_s_abc, double u_dc, double w_M_ref, double *out)
{
    Measurements meas = {abc2complex(i_s_abc), u_dc};
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
    subprocess.run(
        [
            "gcc",
            "-std=c99",
            "-O2",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wno-unused-function",
            f"-I{SRC_DIR}",
            str(tmp / "capi.c"),
            "-lm",
            "-o",
            str(lib),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(lib))
    dll.test_brentq.restype = ctypes.c_double
    dll.test_brentq.argtypes = [ctypes.c_double] * 3
    return dll


def _arr(values: Any) -> Any:
    return (ctypes.c_double * len(values))(*values)


def none(value: float | None) -> float:
    """Represent None by NaN, as in the C port."""
    return float("nan") if value is None else float(value)


# %%
PAR = {"n_p": 3, "R_s": 3.6, "L_d": 0.036, "L_q": 0.051, "psi_f": 0.545}
CFG: dict[str, Any] = {"i_s_max": 6.5}
SPEED = {"J": 0.015, "alpha_s": 25.0}


def test_brentq() -> None:
    """Brent's method should give the same root as SciPy."""
    dll = compile_library()
    for c in (0.1, 0.5, 1.0, 3.0):
        root_c = dll.test_brentq(0.0, 2.0, c)
        root_py = brentq(lambda x, c=c: np.cos(x) - c * x, 0.0, 2.0)
        assert root_c == root_py, (c, root_c, root_py)


def test_luts() -> None:
    """The lookup tables should match those of the reference generator."""
    dll = compile_library()
    par = model.SynchronousMachinePars(**PAR)
    fvc = control.FluxVectorController(par, control.FluxVectorControllerCfg(**CFG))
    rg = cast(Any, fvc.reference_gen)
    out = (ctypes.c_double * (6 * 16))()
    dll.test_luts(_arr(list(PAR.values())), ctypes.c_double(CFG["i_s_max"]), out)
    luts = np.array(out).reshape(3, 2, 16)
    for i, (f, name) in enumerate(
        [(rg.psi_s_mtpa, "psi_s_mtpa"), (rg.tau_M_cl, "tau_M_cl"), (rg.tau_M_mtpv, "")]
    ):
        # Evaluating the Python LUT at the C sample points must give the C values
        err = np.max(np.abs(f(luts[i, 0]) - luts[i, 1]))
        assert err < 1e-12, (name or "tau_M_mtpv", err)
        print(f"  {name or 'tau_M_mtpv'}: max error {err:.3g}")


def test_control_system() -> None:
    """The control system should give the same outputs as motulator."""
    dll = compile_library()
    par = model.SynchronousMachinePars(**PAR)
    cfg = control.FluxVectorControllerCfg(**CFG)
    ctrl = control.VectorControlSystem(
        control.FluxVectorController(par, cfg), control.SpeedController(**SPEED)
    )
    c = [
        cfg.i_s_max,
        cfg.alpha_tau,
        none(cfg.alpha_psi),
        none(cfg.alpha_i),
        none(None),  # alpha_o: default resolved in C
        none(None),
        none(None),  # k_o: default
        none(cfg.psi_s_min),
        cfg.psi_s_max,
        cfg.k_u,
        cfg.k_mtpv,
        none(cfg.J),
        cfg.T_s,
    ]
    s = [SPEED["J"], SPEED["alpha_s"], none(None), inf]
    dll.init(_arr(list(PAR.values())), _arr(c), _arr(s))

    # Measurement sequence: rotating current vector with a varying amplitude
    rng = np.random.default_rng(0)
    n = 4000
    t = np.arange(n) * cfg.T_s
    i_s_ab = (2 + np.sin(2 * pi * 3 * t)) * np.exp(1j * 2 * pi * 20 * t)
    i_s_ab += 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    w_M_ref = np.where(t > 0.05, 50.0, 0.0)
    ctrl.set_speed_ref(lambda t_: 50.0 if t_ > 0.05 else 0.0)

    out = (ctypes.c_double * 8)()
    err = np.zeros(8)
    for k in range(n):
        i_abc = [
            i_s_ab[k].real,
            0.5 * (-i_s_ab[k].real + np.sqrt(3) * i_s_ab[k].imag),
            0.5 * (-i_s_ab[k].real - np.sqrt(3) * i_s_ab[k].imag),
        ]
        # motulator
        meas = Measurements(abc2complex(i_abc), 540.0)
        fbk = cast(Any, ctrl.get_feedback(meas))
        ref = cast(Any, ctrl.compute_output(fbk))
        ctrl.update(ref, fbk)
        py = [*ref.d_abc, fbk.w_M, fbk.tau_M, ref.tau_M, ref.psi_s, fbk.theta_m]
        # C port
        dll.step(_arr(i_abc), ctypes.c_double(540.0), ctypes.c_double(w_M_ref[k]), out)
        err = np.maximum(err, np.abs(np.array(out) - np.array(py)))
    names = ["d_a", "d_b", "d_c", "w_M", "tau_M", "tau_M_ref", "psi_s_ref", "theta_m"]
    for name, e in zip(names, err, strict=True):
        assert e < 1e-9, (name, e)
    print("  max errors:", dict(zip(names, np.round(err, 16), strict=True)))


# %%
if __name__ == "__main__":
    for test in (test_brentq, test_luts, test_control_system):
        test()
        print(f"{test.__name__}: passed")
