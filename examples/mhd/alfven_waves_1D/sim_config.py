from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # Physical parameters
    Re: float = 500.0  # Reynolds number
    Rm: float = 500.0  # Magnetic Reynolds number
    Ma: float = 1.0  # Alfven Mach number

    # Domain
    L: float = 1.0
    T: float = 30.0

    # Numerical parameters
    N: int = 256
    dealias: float = 3 / 2
    dt: float = 1 / 1024

    # Snapshots: every snapshot_steps steps from t_snapshots_start, on a
    # uniform grid of N_x_out points
    t_snapshots_start: float = 20.0
    snapshot_steps: int = 2
    N_x_out: int = 257
    snapshot_path: str = "data/snapshots.hdf5"

    # Waves at two incommensurate frequencies are driven in at x = 0
    freqs: tuple[float, ...] = (np.sqrt(2) / 2, np.sqrt(3))

    def zm_in(self, t: float) -> float:
        amp = np.sqrt(1.0 / len(self.freqs))
        return float(sum(amp * np.sin(2 * np.pi * f * t) for f in self.freqs))
