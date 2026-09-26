# Exporting motulator drive systems to PLECS

This directory contains a pilot for exporting motulator drive systems to PLECS Standalone.
The control system is ported to C and runs in a C-Script block inside a masked subsystem, and the system model is built from PLECS library components.
The mask parameters are the same as in the motulator API (`SynchronousMachinePars`, `FluxVectorControllerCfg`, and `SpeedController`), including the defaults: an empty parameter (`[]`) corresponds to `None` in motulator.
The derived quantities, such as the gains and the lookup tables of the reference generator, are computed by the C code at the start of the simulation, as in motulator.
Hence, the PLECS model is self-contained: the parameters can be changed in the mask without Python.

## Contents

- `src/`: C port of the motulator control algorithms
  - `common.c`: `PIController`, `SpeedObserver`, `PWM`, root finding (`brentq`), and utility functions
  - `sm_control_loci.c`: `ControlLoci` (MTPA, MTPV, and current-limit loci)
  - `sm_flux_vector.c`: `FluxObserver`, `SpeedFluxObserver`, `ReferenceGenerator`, `FluxTorqueController`, `FluxVectorController`, `SpeedController`, and `VectorControlSystem`
- `motulator_plecs.py`: reads the parameters from motulator objects, writes the PLECS model, and simulates it via the RPC interface
- `ipmsm_2kw_fvc.py`: example (the README example of motulator), which exports the model and compares the PLECS and motulator results
- `test_c_port.py`: compares the C port with motulator without PLECS (requires gcc)
- `ipmsm_2kw_fvc.plecs`: generated PLECS model

## Usage

Start PLECS Standalone and enable the RPC interface (Preferences > General > RPC interface, port 1080).
Then, run from the repository root:

```bash
python plecs/ipmsm_2kw_fvc.py
python plecs/test_c_port.py
```

The script writes `plecs/ipmsm_2kw_fvc.plecs`, simulates the drive system in both motulator and PLECS, and prints the maximum differences.
The model can also be opened and simulated directly in PLECS.
The C-Script block includes the C files from the `src` directory, so keep the directory next to the model file.

## Model structure

- The speed reference and the load torque are Step blocks.
- The C-Script block samples the phase currents, the DC-bus voltage, and the speed reference with the sampling period `cfg.T_s`, and outputs the duty ratios with a computational delay of one sampling period, as in motulator.
- The converter is modeled with averaged phase voltages `d_abc*u_dc` (corresponding to the default ZOH model of motulator).
- The machine is the PLECS permanent-magnet synchronous machine (rotor reference frame model), with the inertia of the mechanical system.
- The parameters of the control system are in the mask of the subsystem `Control system` (double-click the block), on three tabs corresponding to `SynchronousMachinePars`, `FluxVectorControllerCfg`, and `SpeedController`. The controller signals are available as probe signals of the subsystem.
- The parameters of the system model are in the initialization commands of the model (Simulation > Simulation Parameters > Initialization): `machine`, `mechanics`, and `converter`.

## Limitations

Currently supported: synchronous machines without magnetic saturation, sensorless flux-vector control with offline reference generation, speed-control mode, and the averaged converter model.
Other configurations raise `NotImplementedError` in `motulator_plecs.py`.
