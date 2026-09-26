"""
GradNet maps evaluated in double precision.

The GradNet maps of motulator are evaluated in single precision (PyTorch float32).
These subclasses evaluate the same networks in double precision, which is used as the
reference when verifying the C port (which uses double precision).

"""

from typing import Any

import numpy as np
import torch

import motulator.drive.gradnet as gn


class FluxMap64(gn.FluxMap):
    """FluxMap evaluated in double precision (reference for the C port)."""

    def __init__(self, model: Any) -> None:
        super().__init__(model.double())

    def __call__(self, i_s_dq: Any) -> Any:
        x = np.array(i_s_dq, ndmin=1, dtype=np.complex128) / self.in_base
        x = np.concatenate([x, np.conj(x)], axis=0)
        inputs = torch.from_numpy(np.stack((x.real, x.imag), axis=-1))
        with torch.no_grad():
            out = self.model(inputs).numpy()
        y = out[..., 0] + 1j * out[..., 1]
        n = y.size // 2
        y = 0.5 * (y[:n] + np.conj(y[n:])) * self.out_base
        return y[0] if y.size == 1 else y


class CurrentMapWithHarmonics64(gn.CurrentMapWithHarmonics):
    """CurrentMapWithHarmonics evaluated in double precision."""

    def __init__(self, model: Any, k: int = 6) -> None:
        super().__init__(model.double(), k)

    def __call__(self, psi_s_dq: Any, exp_j_theta_m: Any) -> Any:
        k = self.k
        psi, exp_j = np.broadcast_arrays(psi_s_dq, exp_j_theta_m)
        psi = np.array(psi, ndmin=1, dtype=np.complex128) / self.psi_base
        e = np.array(exp_j, ndmin=1, dtype=np.complex128) ** k
        x = np.concatenate([psi, np.conj(psi)])
        ek = np.concatenate([e, np.conj(e)])
        inputs = torch.from_numpy(np.stack((x.real, x.imag, ek.real, ek.imag), -1))
        with torch.no_grad():
            out = self.model(inputs).numpy()
        n = psi.size
        i = out[:, 0] + 1j * out[:, 1]
        i = 0.5 * (i[:n] + np.conj(i[n:]))
        dw_dcos = 0.5 * (out[:n, 2] + out[n:, 2])
        dw_dsin = 0.5 * (out[:n, 3] - out[n:, 3])
        dw_dtheta = k * (e.real * dw_dsin - e.imag * dw_dcos)
        tau = (np.imag(i * np.conj(psi)) - dw_dtheta) * self.tau_base
        i = i * self.i_base
        return (i[0], tau[0]) if n == 1 else (i, tau)
