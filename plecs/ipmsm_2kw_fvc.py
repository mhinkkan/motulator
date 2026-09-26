"""
2.2-kW IPMSM, sensorless FVC: motulator vs. PLECS
=================================================

This script builds the drive system of the README example in motulator, exports it
to a PLECS model (ipmsm_2kw_fvc.plecs), and, if PLECS Standalone is running with
the RPC interface enabled, simulates both and compares the results.

Run from the repository root:

    python plecs/ipmsm_2kw_fvc.py

"""

# %%
import sys
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

import motulator.drive.control.sm as control
from motulator.common.model import SolverCfg
from motulator.drive import model

sys.path.insert(0, str(Path(__file__).parent))
from motulator_plecs import (  # noqa: E402
    StepSignal,
    sampled_step,
    simulate_plecs,
    write_plecs_model,
)

T_STOP = 1.2
W_M_REF = StepSignal(time=0.1, after=50.0)  # Speed reference (rad/s)
TAU_L = StepSignal(time=0.7, after=10.0)  # Load torque (Nm)
SPEED_CTRL = {"J": 0.015, "alpha_s": 25.0}  # Arguments of SpeedController


# %%
def build_system() -> tuple[model.Drive, control.VectorControlSystem]:
    """Build the drive system of the README example."""
    par = model.SynchronousMachinePars(
        n_p=3, R_s=3.6, L_d=0.036, L_q=0.051, psi_f=0.545
    )
    mdl = model.Drive(
        model.SynchronousMachine(par),
        model.MechanicalSystem(J=0.015),
        model.VoltageSourceConverter(u_dc=540),
    )
    cfg = control.FluxVectorControllerCfg(i_s_max=6.5, sensorless=True)
    ctrl = control.VectorControlSystem(
        control.FluxVectorController(par, cfg), control.SpeedController(**SPEED_CTRL)
    )
    return mdl, ctrl


def step(sig: StepSignal):
    """Step signal as a function of time, as used in motulator."""
    return lambda t: sig.before + (t > sig.time) * (sig.after - sig.before)


# %%
if __name__ == "__main__":
    # Export the PLECS model (before simulating, since simulation changes the states)
    mdl, ctrl = build_system()
    # Speed reference switching at the same sample as in motulator
    T_s = cast(control.FluxVectorController, ctrl.vector_ctrl).cfg.T_s
    w_M_ref = sampled_step(W_M_REF, T_s)
    path = write_plecs_model(
        Path(__file__).with_name("ipmsm_2kw_fvc.plecs"),
        mdl,
        ctrl,
        w_M_ref,
        TAU_L,
        T_STOP,
        SPEED_CTRL,
    )
    print(f"Wrote {path}")

    # Simulate in motulator with tight tolerances
    ctrl.set_speed_ref(step(W_M_REF))
    mdl.mechanics.set_external_load_torque(step(TAU_L))
    sim = model.Simulation(mdl, ctrl, cfg=SolverCfg(rtol=1e-9, atol=1e-9))
    res = sim.simulate(t_stop=T_STOP)
    # Motulator runs to the end of the last sampling period, drop the extra samples
    in_mdl = res.mdl.t <= T_STOP
    in_ctrl = res.ctrl.t + 0.5 * T_s <= T_STOP

    # Simulate in PLECS, with outputs at the motulator solver steps and just after
    # the sampling instants
    t_mdl = res.mdl.t[in_mdl]
    t_ctrl = res.ctrl.t[in_ctrl] + 0.5 * T_s
    t_eval = np.unique(np.concatenate((t_mdl, t_ctrl)))
    try:
        plecs_mdl, plecs_ctrl = simulate_plecs(path, t_eval)
    except ConnectionRefusedError:
        print("PLECS RPC interface not available, skipping the comparison.")
        sys.exit()

    # Compare the machine signals at the motulator solver steps
    k_mdl = np.searchsorted(t_eval, t_mdl)
    i_s_abc = np.array([plecs_mdl[k][k_mdl] for k in ("i_a", "i_b", "i_c")])
    i_s_ab = (2 / 3) * (i_s_abc[0] - 0.5 * (i_s_abc[1] + i_s_abc[2])) + 1j * (
        i_s_abc[1] - i_s_abc[2]
    ) / np.sqrt(3)
    print("Maximum differences (PLECS - motulator):")
    for name, plecs_value, value in [
        ("w_M", plecs_mdl["w_M"][k_mdl], res.mdl.mechanics.w_M[in_mdl]),
        ("tau_M", plecs_mdl["tau_M"][k_mdl], res.mdl.machine.tau_M[in_mdl]),
        ("i_s_ab", i_s_ab, res.mdl.machine.i_s_ab[in_mdl]),
    ]:
        print(f"  mdl.{name}: {np.max(np.abs(plecs_value - value)):.3g}")

    # Compare the controller signals at the sampling instants
    k_ctrl = np.searchsorted(t_eval, t_ctrl)
    for name, value in [
        ("w_M", res.ctrl.fbk.w_M[in_ctrl]),
        ("tau_M", res.ctrl.fbk.tau_M[in_ctrl]),
        ("tau_M_ref", res.ctrl.ref.tau_M[in_ctrl]),
        ("psi_s_ref", res.ctrl.ref.psi_s[in_ctrl]),
    ]:
        err = np.max(np.abs(plecs_ctrl[name][k_ctrl] - value))
        print(f"  ctrl.{name}: {err:.3g}")

    # Plot
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    t = t_mdl
    axs[0].plot(
        t, res.mdl.mechanics.w_M[in_mdl], label=r"$\omega_\mathrm{M}$ (motulator)"
    )
    axs[0].plot(t, plecs_mdl["w_M"][k_mdl], "--", label=r"$\omega_\mathrm{M}$ (PLECS)")
    axs[0].set_ylabel("Speed (rad/s)")
    axs[1].plot(
        t, res.mdl.machine.tau_M[in_mdl], label=r"$\tau_\mathrm{M}$ (motulator)"
    )
    axs[1].plot(t, plecs_mdl["tau_M"][k_mdl], "--", label=r"$\tau_\mathrm{M}$ (PLECS)")
    axs[1].set_ylabel("Torque (Nm)")
    axs[2].plot(t, res.mdl.machine.i_s_ab[in_mdl].real, label=r"$i_\alpha$ (motulator)")
    axs[2].plot(t, i_s_ab.real, "--", label=r"$i_\alpha$ (PLECS)")
    axs[2].set_ylabel("Current (A)")
    axs[2].set_xlabel("Time (s)")
    for ax in axs:
        ax.legend(loc="upper right")
        ax.grid(True)
    fig.tight_layout()
    plt.show()
