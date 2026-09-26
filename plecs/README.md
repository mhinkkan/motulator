# Exporting motulator drive systems to PLECS

This directory contains a pilot for exporting motulator drive systems to PLECS Standalone.
The control system is ported to C and runs in a C-Script block inside a masked subsystem.
The mask parameters are the same as in the motulator API (`SynchronousMachinePars` or `SaturatedSynchronousMachinePars`, `FluxVectorControllerCfg`, and `SpeedController`), including the defaults: an empty parameter (`[]`) corresponds to `None` in motulator.
The derived quantities, such as the gains and the lookup tables of the reference generator, are computed by the C code at the start of the simulation, as in motulator.
Hence, the PLECS model is self-contained: the parameters can be changed in the mask without Python.

The system model is either the PLECS permanent-magnet synchronous machine (for `SynchronousMachinePars`) or a C-Script block with the machine and the mechanical system (for `SpatialSaturatedSynchronousMachinePars` with a GradNet current map with spatial harmonics).

## Contents

- `src/`: C port of the motulator models and control algorithms
  - `common.c`: `PIController`, `SpeedObserver`, `PWM`, root finding (`brentq`), and utility functions
  - `gradnet.c`: GradNet inference (`FluxMap`, `CurrentMap`, and `CurrentMapWithHarmonics`)
  - `sm_parameters.c`: `SynchronousMachinePars` and `SaturatedSynchronousMachinePars` (flux linkage, incremental inductances, and current iteration)
  - `sm_control_loci.c`: `ControlLoci` (MTPA, MTPV, and current-limit loci)
  - `sm_flux_vector.c`: `FluxObserver`, `SpeedFluxObserver`, `ReferenceGenerator`, `FluxTorqueController`, `FluxVectorController`, `SpeedController`, and `VectorControlSystem`
  - `sm_machine.c`: `SynchronousMachine` with `SpatialSaturatedSynchronousMachinePars` and `MechanicalSystem`
- `motulator_plecs.py`: reads the parameters from motulator objects, writes the PLECS model, and simulates it via the RPC interface
- `ipmsm_2kw_fvc.py`: 2.2-kW IPMSM, sensorless FVC (the README example of motulator)
- `pmsyrm_6kw_gn_fvc.py`: 5.6-kW PM-SyRM, GradNet models from FEM data, sensored FVC (the example `examples/drive/gradnet/plot_6kw_pmsyrm_gn_fvc_fem_harm.py`)
- `gradnet64.py`: GradNet maps of motulator evaluated in double precision, used as the reference for the C port
- `test_c_port.py`, `test_c_port_gradnet.py`: compare the C port with motulator without PLECS (require gcc, the latter also PyTorch)
- `ipmsm_2kw_fvc.plecs`, `pmsyrm_6kw_gn_fvc.plecs`: generated PLECS models

## Usage

Start PLECS Standalone and enable the RPC interface (Preferences > General > RPC interface, port 1080).
Then, run from the repository root:

```bash
python plecs/ipmsm_2kw_fvc.py
python plecs/pmsyrm_6kw_gn_fvc.py
python plecs/test_c_port.py
python plecs/test_c_port_gradnet.py
```

The example scripts write the PLECS models, simulate the drive systems in both motulator and PLECS, and print the maximum differences.
The scripts close the model in PLECS before simulating it, since an open model is not reloaded from the file.
The models can also be opened and simulated directly in PLECS.
The C-Script blocks include the C files from the `src` directory, so keep the directory next to the model files.

## Model structure

- The speed reference and the load torque are Step blocks.
- The control system samples the phase currents, the DC-bus voltage, the speed reference, and the rotor angle (used only in the sensored mode) with the sampling period `T_s`, and outputs the duty ratios with a computational delay of one sampling period, as in motulator.
- The converter is modeled with averaged phase voltages `d_abc*u_dc` (corresponding to the default ZOH model of motulator).
- The parameters of the control system are in the mask of the subsystem `Control system` (double-click the block), on three tabs corresponding to the machine model, `FluxVectorControllerCfg`, and `SpeedController`. A GradNet flux map is given as a struct in the model workspace (`est_flux_map`). The controller signals are available as probe signals of the subsystem.
- The parameters of the system model are in the initialization commands of the model (Simulation > Simulation Parameters > Initialization): `machine`, `mechanics`, and `converter`.

## Numerical precision of GradNets

motulator evaluates the GradNets in single precision (PyTorch), while the C port uses double precision.
The incremental inductances of the control system are computed with finite differences (step 1e-3 A), which amplifies the single-precision rounding errors to about 1 %.
The comparisons therefore use the double-precision GradNets of `gradnet64.py` in motulator.

## Limitations

Currently supported: synchronous machines (constant inductances or GradNet models), flux-vector control with offline reference generation in the sensorless or sensored mode, speed-control mode, and the averaged converter model.
Other configurations raise `NotImplementedError` in `motulator_plecs.py`.
