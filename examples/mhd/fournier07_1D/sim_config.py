from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # Physical parameters
    S: float = 1.0
    Pm: float = 1e-3

    # Numerical parameters
    N: int = 300
    dt: float = 2e-3
    T: float = 0.2

    # Domain
    x_min: float = -1.0
    x_max: float = 1.0

    # Boundary conditions
    u_left: float = 0.0
    u_right: float = 0.0
    b_left: float = -1.0
    b_right: float = 1.0

    # Snapshots
    snapshot_times: tuple[float, ...] | None = tuple(np.linspace(0, 0.2, 101))
    snapshot_dir: str = "data/snapshots"

    # Initial conditions
    @staticmethod
    def u_initial(x: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * x) + 0.4 * np.sin(5 * np.pi * x)

    @staticmethod
    def b_initial(x: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * x) + 2 * np.sin(np.pi * (x + 1) / 4)
