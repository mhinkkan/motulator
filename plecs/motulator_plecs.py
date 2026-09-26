"""
Export motulator drive systems to PLECS.

This module converts a motulator drive system (a continuous-time system model and a
discrete-time control system) into a PLECS Standalone model. The control system is a
masked subsystem, whose mask parameters are the same as in the motulator API
(`SynchronousMachinePars` or `SaturatedSynchronousMachinePars`,
`FluxVectorControllerCfg`, and `SpeedController`). The subsystem contains a C-Script
block, which includes the C port of the motulator control algorithms in the `src`
directory. The derived quantities (e.g., the gains and the lookup tables of the
reference generator) are computed by the C code at the start of the simulation, as in
motulator. The parameters of the system model are written to the initialization
commands of the PLECS model.

Currently supported:

- Synchronous machine: `SynchronousMachinePars` (the PLECS permanent-magnet
  synchronous machine is used) or `SpatialSaturatedSynchronousMachinePars` with a
  GradNet current map with spatial harmonics (the machine and the mechanical system
  are modeled in a C-Script block)
- Mechanical system (`MechanicalSystem` without friction)
- Converter with a stiff DC bus (`VoltageSourceConverter`), averaged (ZOH) PWM model,
  and a computational delay of one sampling period
- Flux-vector control (`FluxVectorController`) in the sensorless or sensored mode,
  with `SynchronousMachinePars` or `SaturatedSynchronousMachinePars` (GradNet flux
  map), and a speed controller (`SpeedController`) in `VectorControlSystem`

The PLECS model can be simulated from Python via the XML-RPC interface of PLECS
Standalone (enable it in Preferences > General > RPC interface).

"""

import base64
from dataclasses import dataclass, fields
from math import isinf, pi
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from motulator.common.control._pwm import PWM
from motulator.common.model._pwm import ZOH
from motulator.drive.control._base import VectorControlSystem
from motulator.drive.control._common import SpeedController
from motulator.drive.control._sm_flux_vector import (
    FluxVectorController,
    FluxVectorControllerCfg,
)
from motulator.drive.model import Drive, MechanicalSystem, SynchronousMachine
from motulator.drive.utils._parameters import (
    SaturatedSynchronousMachinePars,
    SpatialSaturatedSynchronousMachinePars,
    SynchronousMachinePars,
)

SRC_DIR = "src"

# Monitored controller signals, in the order of the control system outputs 2...5
CTRL_OUTPUTS = {
    "speed": ["w_M_ref", "w_M"],
    "torque": ["tau_M_ref", "tau_M"],
    "flux": ["psi_s_ref", "psi_s"],
    "misc": ["theta_m", "i_d", "i_q"],
}

# Machine signals in the output port "mdl"
MDL_OUTPUTS = ["i_a", "i_b", "i_c", "w_M", "theta_M", "tau_M"]

# Inputs of the control system
CTRL_INPUTS = ["i_s_abc", "u_dc", "w_M_ref", "theta_M"]

# GradNet parameters, as fields of a workspace struct and in the C-Script parameters
GRADNET_FIELDS = [
    "in_dim",
    "mu_dim",
    "W",
    "b",
    "mu_log",
    "bias",
    "activation",
    "beta_log",
    "p",
    "in_base",
    "out_base",
]


# %%
@dataclass
class StepSignal:
    """Step signal, `after` at `t >= time` and `before` otherwise."""

    time: float
    after: float
    before: float = 0.0


def sampled_step(sig: StepSignal, T_s: float) -> StepSignal:
    """
    Step signal that switches at the same sample as a reference of motulator.

    The controller of motulator evaluates the reference `(t > sig.time)` at its time
    `t`, which is accumulated by adding the sampling period. Due to rounding, the
    switching sample may differ from that of a PLECS Step block at `sig.time`. This
    returns a step switching halfway between the samples, as in motulator.

    """
    t, k = 0.0, 0
    while not t > sig.time:
        t = (t + T_s) % 1e9  # As in ControlSystem.update of motulator
        k += 1
    return StepSignal(time=(k - 0.5) * T_s, after=sig.after, before=sig.before)


@dataclass
class MaskParam:
    """Mask parameter of the control system."""

    variable: str  # Variable name in the mask workspace
    prompt: str  # Prompt in the mask dialog
    tab: str  # Tab in the mask dialog
    target: str  # C assignment target, or "" if handled separately
    required: bool = False  # Whether an empty value is an error


# Probe signals of the control system, in the order of the C-Script outputs
MASK_PROBES = [
    "Duty ratios (d_abc)",
    "Speed (w_M_ref, w_M)",
    "Torque (tau_M_ref, tau_M)",
    "Flux linkage (psi_s_ref, psi_s)",
    "Angle and current (theta_m, i_d, i_q)",
]

TAB_PAR = "Machine model (SynchronousMachinePars)"
TAB_CFG = "Flux-vector control (FluxVectorControllerCfg)"
TAB_SPEED = "Speed control (SpeedController)"

MASK_PARAMS = [
    MaskParam("n_p", "n_p: Number of pole pairs", TAB_PAR, "", True),
    MaskParam("R_s", "R_s: Stator resistance (Ω)", TAB_PAR, "", True),
    MaskParam(
        "psi_s_dq_fcn",
        "psi_s_dq_fcn: GradNet flux map (SaturatedSynchronousMachinePars), "
        "[] = constant inductances",
        TAB_PAR,
        "",
    ),
    MaskParam("L_d", "L_d: d-axis inductance (H)", TAB_PAR, ""),
    MaskParam("L_q", "L_q: q-axis inductance (H)", TAB_PAR, ""),
    MaskParam("psi_f", "psi_f: PM-flux linkage (Vs)", TAB_PAR, ""),
    MaskParam("i_s_max", "i_s_max: Maximum stator current (A)", TAB_CFG, "", True),
    MaskParam(
        "alpha_tau",
        "alpha_tau: Torque-control bandwidth (rad/s)",
        TAB_CFG,
        "cfg.alpha_tau",
    ),
    MaskParam(
        "alpha_psi",
        "alpha_psi: Flux-control bandwidth (rad/s), [] = alpha_tau",
        TAB_CFG,
        "cfg.alpha_psi",
    ),
    MaskParam(
        "alpha_i",
        "alpha_i: Integral-action bandwidth (rad/s), [] = alpha_tau",
        TAB_CFG,
        "cfg.alpha_i",
    ),
    MaskParam(
        "alpha_o",
        "alpha_o: Speed estimation pole (rad/s), [] = default",
        TAB_CFG,
        "cfg.alpha_o",
    ),
    MaskParam(
        "k_o",
        "k_o: Observer gain [k0 k1], k_o(w_m) = k0 + k1*abs(w_m), [] = default",
        TAB_CFG,
        "",
    ),
    MaskParam(
        "psi_s_min",
        "psi_s_min: Minimum stator flux (Vs), [] = psi_f",
        TAB_CFG,
        "cfg.psi_s_min",
    ),
    MaskParam(
        "psi_s_max", "psi_s_max: Maximum stator flux (Vs)", TAB_CFG, "cfg.psi_s_max"
    ),
    MaskParam("k_u", "k_u: Voltage utilization factor", TAB_CFG, "cfg.k_u"),
    MaskParam("k_mtpv", "k_mtpv: MTPV margin", TAB_CFG, "cfg.k_mtpv"),
    MaskParam(
        "J", "J: Inertia (kgm²) for the speed observer, [] = not used", TAB_CFG, "cfg.J"
    ),
    MaskParam(
        "sensorless",
        "sensorless: Sensorless mode (1) or sensored mode (0)",
        TAB_CFG,
        "cfg.sensorless",
    ),
    MaskParam("T_s", "T_s: Sampling period (s)", TAB_CFG, "cfg.T_s", True),
    MaskParam("speed_J", "J: Total inertia (kgm²)", TAB_SPEED, "", True),
    MaskParam(
        "speed_alpha_s",
        "alpha_s: Reference-tracking bandwidth (rad/s)",
        TAB_SPEED,
        "",
        True,
    ),
    MaskParam(
        "speed_alpha_i",
        "alpha_i: Integral-action bandwidth (rad/s), [] = alpha_s",
        TAB_SPEED,
        "",
    ),
    MaskParam("speed_tau_M_max", "tau_M_max: Maximum motor torque (Nm)", TAB_SPEED, ""),
]

# Mask initialization: the fields of the GradNet flux map as separate variables
MASK_INIT = (
    "% Fields of the GradNet flux map for the C-Script\n"
    "if isempty(psi_s_dq_fcn)\n"
    + "".join(f"  gn_{f} = [];\n" for f in GRADNET_FIELDS)
    + "else\n"
    + "".join(f"  gn_{f} = psi_s_dq_fcn.{f};\n" for f in GRADNET_FIELDS)
    + "end\n"
)


# %%
def _fit_speed_dependent_gain(k: Callable[[float], float], name: str) -> list[float]:
    """Express the gain as k(w_m) = k0 + k1*abs(w_m), checking the form."""
    k0 = float(k(0.0))
    k1 = (float(k(100.0)) - k0) / 100.0
    for w_m in (-300.0, -50.0, 50.0, 300.0, 1e3):
        if not np.isclose(float(k(w_m)), k0 + k1 * abs(w_m), rtol=1e-12, atol=1e-12):
            raise NotImplementedError(f"Only gains k0 + k1*abs(w_m) supported: {name}")
    return [k0, k1]


def export_gradnet(net_map: Any) -> dict[str, Any]:
    """
    Get the parameters of a GradNet map (`FluxMap` or `CurrentMap`) for the C port.

    Single-module GradNets with the PNormGradient or Softmax activations are
    supported, including the current map with spatial harmonics
    (`CurrentMapWithHarmonics`).

    """
    import motulator.drive.gradnet as gn  # noqa: PLC0415

    model = net_map.model
    if model.num_modules != 1 or model.in_dim not in (2, 4):
        raise NotImplementedError("Only single-module GradNets with 2D or 4D inputs")
    block = model.blocks[0]
    act = block.act
    if isinstance(act, gn.PNormGradient):
        activation, p = 1, act.q + 1
    elif isinstance(act, gn.Softmax):
        activation, p = 2, 0
    else:
        raise NotImplementedError(f"Activation not supported: {type(act).__name__}")

    def arr(t: Any) -> np.ndarray:
        return t.detach().cpu().numpy().astype(float)

    harmonics = isinstance(net_map, gn.CurrentMapWithHarmonics)
    return {
        "in_dim": model.in_dim,
        "mu_dim": model.in_dim - model.non_mu_dim,
        "k": net_map.k if harmonics else 0,
        "W": arr(block.W),  # (embed_dim, in_dim)
        "b": arr(block.b),
        "mu_log": arr(model.mu_log),
        "bias": arr(model.bias),
        "activation": activation,
        "beta_log": float(act.beta_log.item()),
        "p": p,
        "in_base": float(net_map.psi_base if harmonics else net_map.in_base),
        "out_base": float(net_map.i_base if harmonics else net_map.out_base),
    }


def _has_gradnet_plant(mdl: Drive) -> bool:
    """Whether the machine model is a GradNet current map with spatial harmonics."""
    return isinstance(mdl.machine.par, SpatialSaturatedSynchronousMachinePars)


def _check_supported(mdl: Drive, ctrl: VectorControlSystem) -> None:
    """Raise an error if the drive system is not supported."""
    # System model
    par = mdl.machine.par
    if not isinstance(mdl.machine, SynchronousMachine) or not isinstance(
        par, (SynchronousMachinePars, SpatialSaturatedSynchronousMachinePars)
    ):
        raise NotImplementedError(
            "Only SynchronousMachinePars and SpatialSaturatedSynchronousMachinePars"
        )
    if isinstance(par, SpatialSaturatedSynchronousMachinePars):
        export_gradnet(par.magnetic_map_fcn)  # Raises if not supported
    if par.G_c != 0:
        raise NotImplementedError("Core losses not supported")
    if not isinstance(mdl.mechanics, MechanicalSystem) or mdl.mechanics.B_L != 0:
        raise NotImplementedError("Only MechanicalSystem without friction supported")
    if mdl.lc_filter is not None or not isinstance(mdl.pwm, ZOH):
        raise NotImplementedError("LC filter and carrier comparison not supported")
    if len(mdl.delay.data) != 1:
        raise NotImplementedError("Only the computational delay of one sample")
    _check_supported_control(ctrl)


def _check_supported_control(ctrl: VectorControlSystem) -> None:
    """Raise an error if the control system is not supported."""
    fvc = ctrl.vector_ctrl
    if not isinstance(fvc, FluxVectorController):
        raise NotImplementedError("Only FluxVectorController supported")
    if fvc.cfg.online_ref or fvc.cfg.k_f is not None:
        raise NotImplementedError("Only offline references and k_f=None supported")
    if isinstance(fvc.par, SaturatedSynchronousMachinePars):
        if fvc.par.psi_s_dq_fcn is None:
            raise NotImplementedError("A flux map (psi_s_dq_fcn) is required")
        export_gradnet(fvc.par.psi_s_dq_fcn)  # Raises if not supported
    elif not isinstance(fvc.par, SynchronousMachinePars):
        raise NotImplementedError("Machine model of the control system not supported")
    if not isinstance(ctrl.speed_ctrl, SpeedController):
        raise NotImplementedError("Speed-control mode with SpeedController required")
    if not isinstance(ctrl.pwm, PWM) or ctrl.pwm.overmodulation != "MPE":
        raise NotImplementedError("Only the MPE overmodulation supported")
    if ctrl.pwm.k_comp != 1.5:
        raise NotImplementedError("Only k_comp = 1.5 supported")


def export_mask_values(
    ctrl: VectorControlSystem, speed_ctrl_args: dict[str, float]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """
    Get the mask parameter values of the control system in the motulator API.

    Parameters
    ----------
    ctrl : VectorControlSystem
        Discrete-time control system.
    speed_ctrl_args : dict[str, float]
        Arguments of `SpeedController` used in `ctrl` (the controller stores only the
        resulting gains). They are checked against the gains of `ctrl`.

    Returns
    -------
    values : dict[str, Any]
        Mask parameter values, None for the defaults.
    flux_map : dict[str, Any] | None
        GradNet flux map of the control system, None if constant inductances are used.

    """
    fvc = cast(FluxVectorController, ctrl.vector_ctrl)
    cfg: FluxVectorControllerCfg = fvc.cfg

    # The speed controller stores only the gains, check the given arguments
    speed_ctrl = cast(SpeedController, ctrl.speed_ctrl)
    ref = SpeedController(**speed_ctrl_args)
    for attr in ("k_p", "k_t", "alpha_i", "u_max"):
        if getattr(ref, attr) != getattr(speed_ctrl, attr):
            raise ValueError(f"speed_ctrl_args do not match the controller: {attr}")

    # alpha_o is resolved in FluxVectorControllerCfg.__post_init__
    default = FluxVectorControllerCfg(
        **{
            f.name: getattr(cfg, f.name)
            for f in fields(cfg)
            if f.name not in ("alpha_o",) and f.init
        }
    )
    alpha_o = None if cfg.alpha_o == default.alpha_o else cfg.alpha_o
    k_o = None if cfg.k_o is None else _fit_speed_dependent_gain(cfg.k_o, "k_o")

    # Machine model of the control system
    flux_map = None
    if isinstance(fvc.par, SaturatedSynchronousMachinePars):
        flux_map = export_gradnet(fvc.par.psi_s_dq_fcn)
        machine = {"n_p": fvc.par.n_p, "R_s": fvc.par.R_s, "L_d": None, "L_q": None}
        machine |= {"psi_f": None, "psi_s_dq_fcn": "est_flux_map"}
    else:
        par = cast(SynchronousMachinePars, fvc.par)
        machine = {"n_p": par.n_p, "R_s": par.R_s, "L_d": par.L_d, "L_q": par.L_q}
        machine |= {"psi_f": par.psi_f, "psi_s_dq_fcn": None}

    values: dict[str, Any] = {
        **machine,
        "i_s_max": cfg.i_s_max,
        "alpha_tau": cfg.alpha_tau,
        "alpha_psi": cfg.alpha_psi,
        "alpha_i": cfg.alpha_i,
        "alpha_o": alpha_o,
        "k_o": k_o,
        "psi_s_min": cfg.psi_s_min,
        "psi_s_max": cfg.psi_s_max,
        "k_u": cfg.k_u,
        "k_mtpv": cfg.k_mtpv,
        "J": cfg.J,
        "sensorless": int(cfg.sensorless),
        "T_s": cfg.T_s,
        "speed_J": speed_ctrl_args["J"],
        "speed_alpha_s": speed_ctrl_args["alpha_s"],
        "speed_alpha_i": speed_ctrl_args.get("alpha_i"),
        "speed_tau_M_max": speed_ctrl_args.get("tau_M_max", float("inf")),
    }
    return values, flux_map


def export_plant_variables(mdl: Drive) -> list[tuple[str, Any]]:
    """Get the workspace variables of the system model."""
    variables: list[tuple[str, Any]] = []
    if _has_gradnet_plant(mdl):
        par = cast(SpatialSaturatedSynchronousMachinePars, mdl.machine.par)
        g = export_gradnet(par.magnetic_map_fcn)
        variables += [("machine.n_p", par.n_p), ("machine.R_s", par.R_s)]
        variables += [("machine.k", g["k"])]
        variables += [(f"machine.current_map.{f}", g[f]) for f in GRADNET_FIELDS]
    else:
        par = cast(SynchronousMachinePars, mdl.machine.par)
        variables += [
            ("machine.n_p", par.n_p),
            ("machine.R_s", par.R_s),
            ("machine.L_d", par.L_d),
            ("machine.L_q", par.L_q),
            ("machine.psi_f", par.psi_f),
        ]
    variables += [
        ("mechanics.J", cast(MechanicalSystem, mdl.mechanics).J),
        ("converter.u_dc", mdl.converter.u_dc),
    ]
    return variables


# %%
def _fmt(value: Any) -> str:
    """Format a value for the PLECS (Octave) workspace."""
    if value is None:
        return "[]"
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 2:
        rows = "; ".join(" ".join(_fmt(v) for v in row) for row in value)
        return f"[{rows}]"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + " ".join(_fmt(v) for v in np.asarray(value).ravel()) + "]"
    value = float(value)
    if isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


def _q(text: str) -> str:
    """Quote a string for the PLECS file format."""
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + text.replace("\n", "\\n").replace("\t", "\\t") + '"'


Point = tuple[int, int]
Terminal = tuple[str, int]


def _points(points: list[Point], indent: str) -> str:
    if not points:
        return ""
    xy = "; ".join(f"{x}, {y}" for x, y in points)
    return f"{indent}Points        [{xy}]\n"


class _Schematic:
    """
    Minimal writer for the components and connections of a PLECS schematic.

    Connections from the same source terminal are written as branches. The points
    of a connection are the corner points of the wire, excluding the terminals. For
    a branched connection, the trunk points lead from the source to the branching
    point, and the branch points from the branching point to the destination.

    """

    def __init__(self) -> None:
        self.items: list[str] = []
        self.connections: dict[
            tuple[Terminal, str], list[tuple[Terminal, list[Point]]]
        ] = {}
        self.trunks: dict[tuple[Terminal, str], list[Point]] = {}

    def component(
        self,
        typ: str,
        name: str,
        pos: Point,
        params: dict[str, str] | None = None,
        direction: str = "right",
        flipped: bool = False,
        extra: str = "",
        show: bool = True,
        label: str = "south",
        src_component: str | None = None,
        trailer: str = "",
    ) -> None:
        """Add a component."""
        s = f"    Component {{\n      Type          {typ}\n"
        if src_component is not None:
            s += f"      SrcComponent  {_q(src_component)}\n"
        s += (
            f"      Name          {_q(name)}\n"
            f"      Show          {'on' if show else 'off'}\n"
            f"      Position      [{pos[0]}, {pos[1]}]\n"
            f"      Direction     {direction}\n"
            f"      Flipped       {'on' if flipped else 'off'}\n"
            f"      LabelPosition {label}\n"
        )
        s += extra
        for var, val in (params or {}).items():
            s += (
                "      Parameter {\n"
                f"        Variable      {_q(var)}\n"
                f"        Value         {_q(val)}\n"
                "        Show          off\n"
                "      }\n"
            )
        self.items.append(s + trailer + "    }\n")

    def connect(
        self, src: Terminal, dst: Terminal, typ: str, points: list[Point] | None = None
    ) -> None:
        """Add a connection. Connections from the same source become branches."""
        self.connections.setdefault((src, typ), []).append((dst, points or []))

    def trunk(self, src: Terminal, typ: str, points: list[Point]) -> None:
        """Set the trunk points of a branched connection."""
        self.trunks[(src, typ)] = points

    def render(self) -> str:
        """Render the components and connections."""
        text = "".join(self.items)
        for (src, typ), dsts in self.connections.items():
            trunk = self.trunks.get((src, typ), [])
            text += (
                "    Connection {\n"
                f"      Type          {typ}\n"
                f"      SrcComponent  {_q(src[0])}\n"
                f"      SrcTerminal   {src[1]}\n"
            )
            if len(dsts) == 1:
                (dst, points) = dsts[0]
                text += (
                    _points(trunk + points, "      ")
                    + f"      DstComponent  {_q(dst[0])}\n"
                    f"      DstTerminal   {dst[1]}\n"
                )
            else:
                text += _points(trunk, "      ")
                for dst, points in dsts:
                    text += (
                        "      Branch {\n"
                        + _points(points, "        ")
                        + f"        DstComponent  {_q(dst[0])}\n"
                        f"        DstTerminal   {dst[1]}\n"
                        "      }\n"
                    )
            text += "    }\n"
        return text

    def signal(
        self, src: Terminal, dst: Terminal, points: list[Point] | None = None
    ) -> None:
        """Add a signal connection."""
        self.connect(src, dst, "Signal", points)

    def wire(
        self, src: Terminal, dst: Terminal, points: list[Point] | None = None
    ) -> None:
        """Add an electrical connection."""
        self.connect(src, dst, "Wire", points)


def _probe(component: str, signals: list[str], path: str = "") -> str:
    sig = ", ".join(_q(s) for s in signals)
    return (
        "      Probe {\n"
        f"        Component     {_q(component)}\n"
        f"        Path          {_q(path)}\n"
        f"        Signals       {{{sig}}}\n"
        "      }\n"
    )


def _scope(axes: list[tuple[str, str]]) -> str:
    extra = (
        "      Location      [100, 100; 800, 700]\n"
        f'      Axes          "{len(axes)}"\n'
        '      TimeRange     "0"\n'
        '      ScrollingMode "1"\n'
        '      SingleTimeAxis "1"\n'
        '      Open          "0"\n'
        '      Ts            "-1"\n'
        '      SampleLimit   "0"\n'
        '      XAxisLabel    "Time (s)"\n'
        '      ShowLegend    "1"\n'
    )
    for name, label in axes:
        extra += (
            "      Axis {\n"
            f"        Name          {_q(name)}\n"
            "        AutoScale     1\n"
            "        MinValue      0\n"
            "        MaxValue      1\n"
            "        Signals       {}\n"
            "        SignalTypes   [ ]\n"
            f"        AxisLabel     {_q(label)}\n"
            "        Untangle      0\n"
            "        KeepBaseline  off\n"
            "        BaselineValue 0\n"
            "      }\n"
        )
    return extra


def _cscript(
    code: dict[str, str],
    num_inputs: str,
    num_outputs: str,
    parameters: str,
    ts: str,
    num_disc_states: int = 0,
    num_cont_states: int = 0,
    feedthrough: str = "1",
) -> dict[str, str]:
    """Parameters of a C-Script block."""
    return {
        "DialogGeometry": "",
        "NumInputs": num_inputs,
        "NumOutputs": num_outputs,
        "NumContStates": str(num_cont_states),
        "NumDiscStates": str(num_disc_states),
        "NumZCSignals": "0",
        "DirectFeedthrough": feedthrough,
        "Ts": ts,
        "Parameters": parameters,
        "LangStandard": "2",
        "GnuExtensions": "2",
        "RuntimeCheck": "2",
        "Declarations": code.get("Declarations", ""),
        "StartFcn": code.get("StartFcn", ""),
        "OutputFcn": code.get("OutputFcn", ""),
        "UpdateFcn": code.get("UpdateFcn", ""),
        "DerivativeFcn": code.get("DerivativeFcn", ""),
        "TerminateFcn": "",
        "StoreCustomStateFcn": "",
        "RestoreCustomStateFcn": "",
    }


# Common C-Script declarations
C_PARAMS = (
    "#define P(i, j) ParamRealData(i, j)\n"
    "#define PDIM(i) (ParamDim(i, 0) * ParamDim(i, 1))\n"
    "\n"
    "/* Parameter value, or NAN for an empty parameter (None in motulator) */\n"
    "#define PARAM(i) (PDIM(i) > 0 ? P(i, 0) : NAN)\n"
    "\n"
    "/* Read a GradNet from the parameters i0, i0 + 1, ... (GRADNET_FIELDS) */\n"
    "#define READ_GRADNET(net, i0) \\\n"
    "    do { \\\n"
    "        (net).in_dim = (int)P((i0), 0); \\\n"
    "        (net).mu_dim = (int)P((i0) + 1, 0); \\\n"
    "        (net).embed_dim = PDIM((i0) + 3); \\\n"
    "        for (int j_ = 0; j_ < (net).embed_dim; j_++) { \\\n"
    "            for (int k_ = 0; k_ < (net).in_dim; k_++) { \\\n"
    "                (net).W[j_][k_] = P((i0) + 2, j_ + (net).embed_dim * k_); \\\n"
    "            } \\\n"
    "            (net).b[j_] = P((i0) + 3, j_); \\\n"
    "        } \\\n"
    "        for (int k_ = 0; k_ < (net).in_dim; k_++) { \\\n"
    "            (net).mu_log[k_] = (k_ < (net).mu_dim) ? P((i0) + 4, k_) : 0.0; \\\n"
    "            (net).bias[k_] = P((i0) + 5, k_); \\\n"
    "        } \\\n"
    "        (net).activation = (GradNetActivation)(int)P((i0) + 6, 0); \\\n"
    "        (net).beta_log = P((i0) + 7, 0); \\\n"
    "        (net).p = (int)P((i0) + 8, 0); \\\n"
    "        (net).in_base = P((i0) + 9, 0); \\\n"
    "        (net).out_base = P((i0) + 10, 0); \\\n"
    "    } while (0)\n"
)


def _control_cscript_code() -> dict[str, str]:
    """Generate the code sections of the control-system C-Script block."""
    declarations = (
        "/* Generated by motulator_plecs.py. The control algorithms are in the\n"
        " * included C files. The parameters come from the mask of the subsystem. */\n"
        f'#include "{SRC_DIR}/common.c"\n'
        f'#include "{SRC_DIR}/gradnet.c"\n'
        f'#include "{SRC_DIR}/sm_parameters.c"\n'
        f'#include "{SRC_DIR}/sm_control_loci.c"\n'
        f'#include "{SRC_DIR}/sm_flux_vector.c"\n'
        "\n" + C_PARAMS + "\n"
        "static VectorControlSystem ctrl;\n"
    )
    i = {m.variable: k for k, m in enumerate(MASK_PARAMS)}
    i_gn = len(MASK_PARAMS)  # The GradNet parameters follow the mask parameters
    checks = "".join(
        f"if (PDIM({i[m.variable]}) != 1) {{\n"
        f'    SetErrorMessage("{m.variable} must be a scalar.");\n'
        "    return;\n"
        "}\n"
        for m in MASK_PARAMS
        if m.required
    )
    cfg = "".join(
        f"if (PDIM({i[m.variable]}) > 0) {{\n"
        f"    {m.target} = P({i[m.variable]}, 0);\n"
        "}\n"
        for m in MASK_PARAMS
        if m.target.startswith("cfg.")
    )
    k_o, tau_M_max = i["k_o"], i["speed_tau_M_max"]
    L_d, L_q, psi_f = i["L_d"], i["L_q"], i["psi_f"]
    start = (
        "/* Check the parameters of the mask */\n"
        + checks
        + f"if (PDIM({k_o}) != 0 && PDIM({k_o}) != 2) {{\n"
        '    SetErrorMessage("k_o must be [] or [k0 k1].");\n'
        "    return;\n"
        "}\n"
        "\n"
        "/* Machine model parameters: SaturatedSynchronousMachinePars with a GradNet\n"
        " * flux map, or SynchronousMachinePars with constant inductances */\n"
        "SynchronousMachinePars par;\n"
        f"if (PDIM({i_gn}) > 0) {{\n"
        "    GradNet flux_map;\n"
        f"    READ_GRADNET(flux_map, {i_gn});\n"
        "    par = saturated_synchronous_machine_pars(\n"
        f"        P({i['n_p']}, 0), P({i['R_s']}, 0), &flux_map);\n"
        "} else {\n"
        f"    if (PDIM({L_d}) != 1 || PDIM({L_q}) != 1 || PDIM({psi_f}) != 1) {{\n"
        '        SetErrorMessage("L_d, L_q, and psi_f needed without a flux map.");\n'
        "        return;\n"
        "    }\n"
        "    par = synchronous_machine_pars(\n"
        f"        P({i['n_p']}, 0), P({i['R_s']}, 0), P({L_d}, 0), P({L_q}, 0),\n"
        f"        P({psi_f}, 0));\n"
        "}\n"
        "\n"
        "/* Flux-vector controller configuration (FluxVectorControllerCfg), the\n"
        " * defaults are used for empty parameters */\n"
        "FluxVectorControllerCfg cfg = "
        f"flux_vector_controller_cfg(P({i['i_s_max']}, 0));\n"
        + cfg
        + f"if (PDIM({k_o}) == 2) {{\n"
        f"    cfg.k_o[0] = P({k_o}, 0);\n"
        f"    cfg.k_o[1] = P({k_o}, 1);\n"
        "}\n"
        "\n"
        "/* Speed controller (SpeedController) */\n"
        "PIController speed_ctrl = speed_controller(\n"
        f"    P({i['speed_J']}, 0), P({i['speed_alpha_s']}, 0), "
        f"PARAM({i['speed_alpha_i']}),\n"
        f"    PDIM({tau_M_max}) > 0 ? P({tau_M_max}, 0) : INFINITY);\n"
        "\n"
        "vector_control_system_init(&ctrl, par, &cfg, speed_ctrl);\n"
        "\n"
        "/* Duty ratios delayed by one sampling period (computational delay) */\n"
        "for (int k = 0; k < 3; k++) {\n"
        "    DiscState(k) = 0.0;\n"
        "}\n"
    )
    monitored = {
        "w_M_ref": "ctrl.ref.w_M",
        "w_M": "ctrl.fbk.w_M",
        "tau_M_ref": "ctrl.ref.tau_M",
        "tau_M": "ctrl.fbk.tau_M",
        "psi_s_ref": "ctrl.ref.psi_s",
        "psi_s": "cabs(ctrl.fbk.psi_s)",
        "theta_m": "ctrl.fbk.theta_m",
        "i_d": "creal(ctrl.fbk.i_s)",
        "i_q": "cimag(ctrl.fbk.i_s)",
    }
    output = (
        "/* Measurements */\n"
        "double i_s_abc[3] = {InputSignal(0, 0), InputSignal(0, 1),\n"
        "                     InputSignal(0, 2)};\n"
        "Measurements meas = {abc2complex(i_s_abc), InputSignal(1, 0),\n"
        "                     InputSignal(3, 0)};\n"
        "double w_M_ref = InputSignal(2, 0);\n"
        "\n"
        "vector_control_system_compute_output(&ctrl, &meas, w_M_ref);\n"
        "\n"
        "/* Duty ratios computed during the previous sampling period */\n"
        "for (int k = 0; k < 3; k++) {\n"
        "    OutputSignal(0, k) = DiscState(k);\n"
        "}\n"
        "\n"
        "/* Monitored signals */\n"
    )
    for i_out, names in enumerate(CTRL_OUTPUTS.values()):
        for j, name in enumerate(names):
            output += f"OutputSignal({i_out + 1}, {j}) = {monitored[name]};\n"
    update = (
        "vector_control_system_update(&ctrl);\n"
        "for (int k = 0; k < 3; k++) {\n"
        "    DiscState(k) = ctrl.ref.d_abc[k];\n"
        "}\n"
    )
    return {
        "Declarations": declarations,
        "StartFcn": start,
        "OutputFcn": output,
        "UpdateFcn": update,
    }


def _machine_cscript_code() -> dict[str, str]:
    """Generate the code sections of the machine C-Script block (GradNet plant)."""
    declarations = (
        "/* Generated by motulator_plecs.py. The machine model is in the included C\n"
        " * files. The parameters come from the initialization commands. */\n"
        f'#include "{SRC_DIR}/common.c"\n'
        f'#include "{SRC_DIR}/gradnet.c"\n'
        f'#include "{SRC_DIR}/sm_machine.c"\n'
        "\n" + C_PARAMS + "\n"
        "static SpatialSaturatedSynchronousMachinePars par;\n"
        "static double J;\n"
        "\n"
        "/* Continuous states */\n"
        "#define PSI_D ContState(0)\n"
        "#define PSI_Q ContState(1)\n"
        "#define THETA_M ContState(2)\n"
        "#define W_M ContState(3)\n"
    )
    start = (
        "GradNet current_map;\n"
        "READ_GRADNET(current_map, 3);\n"
        "par = spatial_saturated_synchronous_machine_pars(\n"
        "    P(0, 0), P(1, 0), &current_map, (int)P(2, 0));\n"
        f"J = P({3 + len(GRADNET_FIELDS)}, 0);\n"
        "\n"
        "/* Initial states as in motulator */\n"
        "PSI_D = par.psi_f;\n"
        "PSI_Q = 0.0;\n"
        "THETA_M = 0.0;\n"
        "W_M = 0.0;\n"
    )
    output = (
        "double complex i_s_dq;\n"
        "double tau_M;\n"
        "double theta_m = par.n_p * THETA_M;\n"
        "machine_magnetic_map(&par, PSI_D + I * PSI_Q, theta_m, &i_s_dq, &tau_M);\n"
        "double i_s_abc[3];\n"
        "complex2abc(i_s_dq * cexp(I * theta_m), i_s_abc);\n"
        "for (int k = 0; k < 3; k++) {\n"
        "    OutputSignal(0, k) = i_s_abc[k];\n"
        "}\n"
        "OutputSignal(1, 0) = W_M;\n"
        "OutputSignal(2, 0) = THETA_M;\n"
        "OutputSignal(3, 0) = tau_M;\n"
    )
    derivative = (
        "double u_s_abc[3] = {InputSignal(0, 0), InputSignal(0, 1),\n"
        "                     InputSignal(0, 2)};\n"
        "double complex d_psi_s_dq;\n"
        "double d_theta_M, d_w_M;\n"
        "machine_rhs(&par, J, abc2complex(u_s_abc), InputSignal(1, 0),\n"
        "            PSI_D + I * PSI_Q, THETA_M, W_M, &d_psi_s_dq, &d_theta_M,\n"
        "            &d_w_M);\n"
        "ContDeriv(0) = creal(d_psi_s_dq);\n"
        "ContDeriv(1) = cimag(d_psi_s_dq);\n"
        "ContDeriv(2) = d_theta_M;\n"
        "ContDeriv(3) = d_w_M;\n"
    )
    return {
        "Declarations": declarations,
        "StartFcn": start,
        "OutputFcn": output,
        "DerivativeFcn": derivative,
    }


def _step(sig: StepSignal) -> dict[str, str]:
    return {
        "Time": _fmt(sig.time),
        "Before": _fmt(sig.before),
        "After": _fmt(sig.after),
        "DataType": "10",
    }


# Layout of the schematic (x to the right, y downwards). The control system is on
# the left, the converter and the machine on the right, and the scope at the bottom.
CS = (300, 200)  # Control system; inputs at x - 65, outputs at x + 55
N_IN = len(CTRL_INPUTS)  # The outputs of the control system start from N_IN + 1


def _fmt_mask(value: Any) -> str:
    """Format a mask value, writing multiples of 2*pi as in motulator."""
    if isinstance(value, float) and value > 0 and not isinf(value):
        k = value / (2 * pi)
        if k == round(k, 6) and 2 * pi * round(k, 6) == value:
            return f"2*pi*{round(k, 6):g}"
    return _fmt(value)


def _mask_parameter(m: MaskParam, value: Any) -> str:
    """Mask parameter definition of a subsystem."""
    prompt = (
        _q(m.prompt)
        if m.prompt.isascii()
        else "base64 " + _q(base64.b64encode(m.prompt.encode()).decode())
    )
    return (
        "      Parameter {\n"
        f"        Variable      {_q(m.variable)}\n"
        f"        Prompt        {prompt}\n"
        "        Type          FreeText\n"
        f"        Value         {_q(_fmt_mask(value))}\n"
        "        Show          off\n"
        # Tunable, since non-tunable parameters are inlined as constants, which
        # makes accessing an empty parameter (None) a compilation error
        "        Tunable       on\n"
        f"        TabName       {_q(m.tab)}\n"
        "      }\n"
    )


def _control_subsystem(mask_values: dict[str, Any]) -> tuple[str, str]:
    """Mask and contents of the control-system subsystem."""
    n_ctrl = [len(v) for v in CTRL_OUTPUTS.values()]
    header = (
        "      Frame         [-50, -60; 50, 60]\n"
        '      SampleTime    "-1"\n'
        '      CodeGenDiscretizationMethod "2"\n'
        '      CodeGenTarget "Generic"\n'
        '      MaskType      "Flux-vector control (motulator)"\n'
        '      MaskDescription "Speed control of a synchronous machine drive with '
        "flux-vector control. The parameters correspond to the motulator API: "
        "SynchronousMachinePars (or SaturatedSynchronousMachinePars with a GradNet "
        "flux map), FluxVectorControllerCfg, and SpeedController. Empty parameters "
        '([]) correspond to None, i.e., the defaults of motulator."\n'
        f"      MaskInit      {_q(MASK_INIT)}\n"
        '      MaskDisplayLang "2"\n'
        "      MaskDisplay   "
        + _q("Icon:text(0, -8, 'Flux-vector')\nIcon:text(0, 8, 'control')")
        + "\n"
        "      MaskIconFrame on\n"
        "      MaskIconOpaque off\n"
        "      MaskIconRotates on\n"
        + "".join(_mask_parameter(m, mask_values[m.variable]) for m in MASK_PARAMS)
    )
    outputs = ["d_abc", *CTRL_OUTPUTS]
    terminals = "".join(
        "      Terminal {\n"
        f"        Type          {typ}\n"
        f"        Position      [{x}, {y}]\n"
        f"        Direction     {d}\n"
        "      }\n"
        for typ, x, y, d in [
            *[("Input", -50, 10 * k - 10, "left") for k in range(N_IN)],
            *[("Output", 54, 10 * k - 20, "right") for k in range(len(outputs))],
        ]
    )

    # Contents: input ports, C-Script, and output ports
    sub = _Schematic()
    y_cs = 90  # C-Script position, inputs spaced by 10 around it
    for k, name in enumerate(CTRL_INPUTS):
        y_in = y_cs + 10 * k - 5 * (N_IN - 1)
        sub.component(
            "Input", name, (60, 40 + 30 * k), {"Index": str(k + 1), "Width": "-1"}
        )
        x = 100 + 8 * k  # Separate vertical segments for each input
        sub.signal((name, 1), ("C-Script", k + 1), [(x, 40 + 30 * k), (x, y_in)])
    # The flux map (a struct) is passed via the gn_* variables of the mask
    # initialization, since the C-Script parameters must be numeric
    parameters = [
        f"isempty({m.variable})" if m.variable == "psi_s_dq_fcn" else m.variable
        for m in MASK_PARAMS
    ]
    parameters += [f"gn_{f}" for f in GRADNET_FIELDS]
    cscript = _cscript(
        _control_cscript_code(),
        "[3 1 1 1]",
        "[3 " + " ".join(str(n) for n in n_ctrl) + "]",
        ", ".join(parameters),
        "T_s",
        num_disc_states=3,
    )
    sub.component(
        "CScript",
        "C-Script",
        (200, y_cs),
        cscript,
        direction="up",
        extra="      Frame         [-50, -60; 50, 60]\n",
    )
    for k, name in enumerate(outputs):
        y = 20 + 35 * k
        # The port index runs over both the inputs and the outputs
        index = str(N_IN + k + 1)
        sub.component("Output", name, (340, y), {"Index": index, "Width": "-1"})
        y_out = y_cs + 10 * k - 5 * (len(outputs) - 1)
        x = 265 + 8 * k  # Separate vertical segments for each output
        sub.signal(("C-Script", N_IN + 1 + k), (name, 1), [(x, y_out), (x, y)])
    schematic = (
        "      Schematic {\n"
        "        Location      [0, 0; 500, 250]\n"
        "        ZoomFactor    1\n"
        "        SliderPosition [0, 0]\n"
        "        ShowBrowser   off\n"
        "        BrowserWidth  100\n" + sub.render() + "      }\n"
    )
    # Probe signals of the masked subsystem (the internals cannot be probed directly)
    mask_probes = "".join(
        "      MaskProbe {\n"
        f"        Name          {_q(name)}\n"
        "        Probe {\n"
        '          Component     "C-Script"\n'
        '          Path          ""\n'
        f'          Signals       {{"Output {k + 1}"}}\n'
        "        }\n"
        "      }\n"
        for k, name in enumerate(MASK_PROBES)
    )
    return header, terminals + schematic + mask_probes


# Probe signals of the machine: (current, speed, angle, torque)
PMSM_PROBES = ["Stator phase currents", "Rotational speed", "Rotor position"]
PMSM_PROBES += ["Electrical torque"]
GRADNET_PROBES = ["Output 1", "Output 2", "Output 3", "Output 4"]


def _add_control_system(
    sch: _Schematic,
    mask_values: dict[str, Any],
    w_M_ref: StepSignal,
    machine_probes: list[str],
) -> None:
    """Add the references, measurements, and the control system."""
    # Measurements and references
    sch.component(
        "PlecsProbe", "i_s_abc", (110, 130), extra=_probe("Machine", machine_probes[:1])
    )
    sch.signal(("i_s_abc", 1), ("Control system", 1), [(210, 130), (210, CS[1] - 10)])
    sch.component(
        "Constant", "u_dc", (110, 200), {"Value": "converter.u_dc", "DataType": "10"}
    )
    sch.signal(("u_dc", 1), ("Control system", 2))
    sch.component("Step", "w_M_ref", (110, 270), _step(w_M_ref))
    sch.signal(("w_M_ref", 1), ("Control system", 3), [(210, 270), (210, CS[1] + 10)])
    sch.component(
        "PlecsProbe",
        "theta_M",
        (110, 330),
        extra=_probe("Machine", machine_probes[2:3]),
    )
    sch.signal(("theta_M", 1), ("Control system", 4), [(220, 330), (220, CS[1] + 20)])

    # Control system as a masked subsystem
    header, trailer = _control_subsystem(mask_values)
    sch.component(
        "Subsystem", "Control system", CS, direction="up", extra=header, trailer=trailer
    )


def _add_converter(sch: _Schematic) -> None:
    """Add the converter (averaged model): phase voltages d_abc*u_dc."""
    sch.component(
        "Gain",
        "Converter",
        (410, 180),
        {"K": "converter.u_dc", "Multiplication": "1", "DataType": "11"},
    )
    sch.signal(("Control system", N_IN + 1), ("Converter", 1))


def _add_pmsm_plant(sch: _Schematic, tau_L: StepSignal) -> None:
    """Add the PLECS permanent-magnet synchronous machine and mechanics."""
    sch.component(
        "SignalDemux", "Demux", (470, 180), {"Width": "[1 1 1]"}, flipped=True
    )
    sch.signal(("Converter", 2), ("Demux", 1))
    sch.component("Ground", "Ground", (620, 150), direction="up", show=False)
    sch.trunk(("Ground", 1), "Wire", [(620, 130)])
    for k, ph in enumerate("abc"):
        x = 560 + 60 * k
        sch.component(
            "VoltageSource",
            f"u_{ph}",
            (x, 100),
            {"DiscretizationBehavior": "2", "StateSpaceInlining": "1"},
            direction="down",
            flipped=True,
            label="east",
        )
        # Duty ratio to the control input on the left side of the source
        y_d = 170 + 10 * k
        sch.signal(("Demux", k + 2), (f"u_{ph}", 3), [(x - 35, y_d), (x - 35, 100)])
        # Negative terminal to ground, positive terminal to the machine
        sch.wire(("Ground", 1), (f"u_{ph}", 2), [] if x == 620 else [(x, 130)])
        y_m, x_m = 50 + 10 * k, 710 - 10 * k
        sch.wire(
            (f"u_{ph}", 1),
            ("Machine", k + 1),
            [(x, y_m), (x_m, y_m), (x_m, 80 + 10 * k)],
        )

    # Machine and mechanics
    machine = {
        "configuration": "1",
        "R": "machine.R_s",
        "L": "[machine.L_d machine.L_q]",
        "phi": "machine.psi_f",
        "J": "mechanics.J",
        "F": "0",
        "p": "machine.n_p",
        "wm0": "0",
        "thm0": "0",
        "is0": "[0 0]",
    }
    terminals = "".join(
        "      Terminal {\n"
        f"        Type          {typ}\n"
        f"        Position      [{x}, {y}]\n"
        f"        Direction     {d}\n"
        "      }\n"
        for typ, x, y, d in [
            ("Port", -30, -10, "left"),
            ("Port", -30, 0, "left"),
            ("Port", -30, 10, "left"),
            ("Rotational", 30, 30, "right"),
        ]
    )
    sch.component(
        "Reference",
        "Machine",
        (760, 90),
        machine,
        direction="up",
        label="east",
        src_component="Components/Electrical/Machines/Perm.-Magnet SM",
        extra="      Frame         [-25, -25; 25, 35]\n",
        trailer=terminals,
    )
    sch.component("Step", "tau_L", (710, 170), _step(tau_L))
    sch.component(
        "ControlledTorque",
        "Load",
        (790, 170),
        {"SecondFlange": "2", "StateSpaceInlining": "2"},
        direction="left",
        label="east",
    )
    sch.component(
        "RotationalReference", "Frame", (790, 215), direction="up", show=False
    )
    sch.connect(("Machine", 4), ("Load", 3), "Rotational")
    sch.connect(("Frame", 1), ("Load", 1), "Rotational")
    sch.signal(("tau_L", 1), ("Load", 2))

    # Output port for scripted simulations
    sch.component(
        "PlecsProbe",
        "Machine signals",
        (560, 300),
        extra=_probe("Machine", PMSM_PROBES),
    )
    sch.component("Output", "mdl", (650, 300), {"Index": "1", "Width": "-1"})
    sch.signal(("Machine signals", 1), ("mdl", 1))


def _add_gradnet_plant(sch: _Schematic, tau_L: StepSignal) -> None:
    """Add the machine with a GradNet current map and the mechanics (C-Script)."""
    parameters = ["machine.n_p", "machine.R_s", "machine.k"]
    parameters += [f"machine.current_map.{f}" for f in GRADNET_FIELDS]
    parameters += ["mechanics.J"]
    cscript = _cscript(
        _machine_cscript_code(),
        "[3 1]",
        "[3 1 1 1]",
        ", ".join(parameters),
        "0",
        num_cont_states=4,
        feedthrough="0",
    )
    sch.component(
        "CScript",
        "Machine",
        (620, 185),
        cscript,
        direction="up",
        label="north",
        extra="      Frame         [-50, -40; 50, 40]\n",
    )
    sch.signal(("Converter", 2), ("Machine", 1))
    sch.component("Step", "tau_L", (520, 240), _step(tau_L))
    sch.signal(("tau_L", 1), ("Machine", 2), [(545, 240), (545, 190)])

    # Output port for scripted simulations: [i_s_abc, w_M, theta_M, tau_M]
    sch.component(
        "SignalMux", "Mux mdl", (720, 185), {"Width": "[3 1 1 1]"}, show=False
    )
    for k in range(4):
        sch.signal(("Machine", 3 + k), ("Mux mdl", 2 + k))
    sch.component("Output", "mdl", (790, 185), {"Index": "1", "Width": "-1"})
    sch.signal(("Mux mdl", 1), ("mdl", 1))


def _add_outputs(sch: _Schematic, machine_probes: list[str]) -> None:
    """Add the output port of the controller signals and the scope."""
    n_ctrl = [len(v) for v in CTRL_OUTPUTS.values()]

    # Output port for scripted simulations: controller signals
    sch.component(
        "SignalMux",
        "Mux",
        (420, 205),
        {"Width": "[" + " ".join(str(n) for n in n_ctrl) + "]"},
        show=False,
    )
    for i in range(len(n_ctrl)):
        sch.signal(("Control system", N_IN + 2 + i), ("Mux", i + 2))
    sch.component("Output", "ctrl", (480, 250), {"Index": "2", "Width": "-1"})
    sch.signal(("Mux", 1), ("ctrl", 1), [(440, 205), (440, 250)])

    # Scope, fed by probes of the controller outputs and the machine
    probes = [
        ("Speed ref. & est.", "Control system", [MASK_PROBES[1]], (720, 310)),
        ("Speed", "Machine", [machine_probes[1]], (720, 350)),
        ("Torque ref. & est.", "Control system", [MASK_PROBES[2]], (720, 385)),
        ("Torque", "Machine", [machine_probes[3]], (720, 420)),
        ("Currents", "Machine", [machine_probes[0]], (720, 460)),
        ("Flux ref. & est.", "Control system", [MASK_PROBES[3]], (720, 500)),
    ]
    for name, comp, signals, pos in probes:
        sch.component(
            "PlecsProbe", name, pos, extra=_probe(comp, signals), label="north"
        )
    sch.component("SignalMux", "Mux speed", (800, 330), {"Width": "[2 1]"}, show=False)
    sch.signal(("Speed ref. & est.", 1), ("Mux speed", 2), [(760, 310), (760, 325)])
    sch.signal(("Speed", 1), ("Mux speed", 3), [(770, 350), (770, 335)])
    sch.component("SignalMux", "Mux torque", (800, 400), {"Width": "[2 1]"}, show=False)
    sch.signal(("Torque ref. & est.", 1), ("Mux torque", 2), [(765, 385), (765, 395)])
    sch.signal(("Torque", 1), ("Mux torque", 3), [(765, 420), (765, 405)])
    sch.component(
        "Scope",
        "Scope",
        (900, 380),
        extra=_scope(
            [
                ("Speed", "Speed (rad/s)"),
                ("Torque", "Torque (Nm)"),
                ("Current", "Current (A)"),
                ("Flux linkage", "Flux linkage (Vs)"),
            ]
        ),
        direction="up",
    )
    sch.signal(("Mux speed", 1), ("Scope", 1), [(850, 330), (850, 365)])
    sch.signal(("Mux torque", 1), ("Scope", 2), [(855, 400), (855, 375)])
    sch.signal(("Currents", 1), ("Scope", 3), [(860, 460), (860, 385)])
    sch.signal(("Flux ref. & est.", 1), ("Scope", 4), [(865, 500), (865, 395)])


def write_plecs_model(
    path: str | Path,
    mdl: Drive,
    ctrl: VectorControlSystem,
    w_M_ref: StepSignal,
    tau_L: StepSignal,
    t_stop: float,
    speed_ctrl_args: dict[str, float],
) -> Path:
    """
    Write a PLECS model of the drive system.

    Parameters
    ----------
    path : str | Path
        Path of the model file (.plecs). The C sources are expected in the `src`
        directory next to the model file.
    mdl : Drive
        Continuous-time system model.
    ctrl : VectorControlSystem
        Discrete-time control system.
    w_M_ref : StepSignal
        Speed reference (mechanical rad/s).
    tau_L : StepSignal
        External load torque (Nm).
    t_stop : float
        Simulation stop time (s).
    speed_ctrl_args : dict[str, float]
        Arguments of `SpeedController` used in `ctrl`, e.g., ``{"J": 0.015,
        "alpha_s": 25}``.

    Returns
    -------
    Path
        Path of the written model file.

    """
    path = Path(path)
    _check_supported(mdl, ctrl)
    mask_values, flux_map = export_mask_values(ctrl, speed_ctrl_args)
    variables = export_plant_variables(mdl)
    if flux_map is not None:
        variables += [(f"est_flux_map.{f}", flux_map[f]) for f in GRADNET_FIELDS]
    init = (
        "% Generated by motulator_plecs.py. The parameters of the control system\n"
        "% are in the mask of the subsystem 'Control system'.\n"
        + "".join(f"{name} = {_fmt(value)};\n" for name, value in variables)
    )

    sch = _Schematic()
    gradnet_plant = _has_gradnet_plant(mdl)
    machine_probes = GRADNET_PROBES if gradnet_plant else PMSM_PROBES
    _add_control_system(sch, mask_values, w_M_ref, machine_probes)
    _add_converter(sch)
    if gradnet_plant:
        _add_gradnet_plant(sch, tau_L)
    else:
        _add_pmsm_plant(sch, tau_L)
    _add_outputs(sch, machine_probes)

    text = (
        "Plecs {\n"
        f"  Name          {_q(path.stem)}\n"
        '  Version       "5.0"\n'
        '  CircuitModel  "ContStateSpace"\n'
        '  StartTime     "0.0"\n'
        f"  TimeSpan      {_q(_fmt(t_stop))}\n"
        '  Solver        "dopri"\n'
        f"  MaxStep       {_q(_fmt(mask_values['T_s']))}\n"
        '  InitStep      "-1"\n'
        '  RelTol        "1e-6"\n'
        '  AbsTol        "-1"\n'
        f"  InitializationCommands {_q(init)}\n"
        '  Terminal {\n    Type          Output\n    Index         "1"\n  }\n'
        '  Terminal {\n    Type          Output\n    Index         "2"\n  }\n'
        "  Schematic {\n"
        "    Location      [0, 0; 1000, 500]\n"
        "    ZoomFactor    1\n"
        "    SliderPosition [0, 0]\n"
        "    ShowBrowser   off\n"
        "    BrowserWidth  100\n" + sch.render() + "  }\n}\n"
    )
    path.write_text(text)
    return path


# %%
def simulate_plecs(
    path: str | Path,
    t_eval: np.ndarray,
    model_vars: dict[str, float] | None = None,
    url: str = "http://localhost:1080/RPC2",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Simulate the PLECS model via the XML-RPC interface of PLECS Standalone.

    Parameters
    ----------
    path : str | Path
        Path of the model file.
    t_eval : ndarray
        Output times (s).
    model_vars : dict[str, float], optional
        Workspace variables to be overridden.
    url : str, optional
        URL of the RPC interface, defaults to "http://localhost:1080/RPC2".

    Returns
    -------
    mdl : dict[str, ndarray]
        Machine signals, see `MDL_OUTPUTS`, and the time "t".
    ctrl : dict[str, ndarray]
        Controller signals, see `CTRL_OUTPUTS`, and the time "t".

    """
    import xmlrpc.client  # noqa: PLC0415

    path = Path(path).resolve()
    server = xmlrpc.client.ServerProxy(url)
    # Close the model first, since loading a model that is already open (e.g., in
    # the PLECS window) does not reload it from the file
    try:
        server.plecs.close(path.stem)
    except xmlrpc.client.Fault:
        pass
    server.plecs.load(str(path))
    opts: dict[str, Any] = {"SolverOpts": {"OutputTimes": [float(t) for t in t_eval]}}
    if model_vars:
        opts["ModelVars"] = model_vars
    try:
        res: Any = server.plecs.simulate(path.stem, opts)
    finally:
        server.plecs.close(path.stem)
    t = np.array(res["Time"])
    values = np.array(res["Values"])
    ctrl_names = [n for v in CTRL_OUTPUTS.values() for n in v]
    if values.shape[0] != len(MDL_OUTPUTS) + len(ctrl_names):
        raise RuntimeError(f"Unexpected number of output signals: {values.shape}")
    mdl = {"t": t, **dict(zip(MDL_OUTPUTS, values[: len(MDL_OUTPUTS)], strict=True))}
    ctrl = {"t": t, **dict(zip(ctrl_names, values[len(MDL_OUTPUTS) :], strict=True))}
    return mdl, ctrl
