from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from numpy.polynomial.legendre import Legendre
from sim_config import Config


class MHDSolver1D:
    def __init__(self, cfg: Config):
        # Config
        self.cfg = cfg

        # GLL grid
        self.x_gll, self.w_gll = self.gll_points_weights(cfg.N)

        # Scale to domain
        self.x = 0.5 * (cfg.x_max - cfg.x_min) * self.x_gll + 0.5 * (
            cfg.x_max + cfg.x_min
        )
        self.w = self.w_gll * (cfg.x_max - cfg.x_min) / 2

        # Derivative matrix
        D_ref = self.derivative_matrix(self.x_gll)
        scale = 2 / (cfg.x_max - cfg.x_min)
        self.D = scale * D_ref

        # Mass matrix
        self.M = np.diag(self.w)

        # Stiffness matrix
        self.K = self.D.T @ self.M @ self.D

        # Helmholtz operators (Fournier+2007 Eqs 24, 25)
        self.H_u = self.M / cfg.dt + cfg.Pm * self.K
        self.H_b = self.M / cfg.dt + self.K

        # Set ICs
        self.u = self.cfg.u_initial(self.x)
        self.b = self.cfg.b_initial(self.x)

        # Enforce BCs
        self.u[0] = cfg.u_left
        self.u[-1] = cfg.u_right
        self.b[0] = cfg.b_left
        self.b[-1] = cfg.b_right
        for H in [self.H_u, self.H_b]:
            H[0, :] = 0.0
            H[0, 0] = 1.0
            H[-1, :] = 0.0
            H[-1, -1] = 1.0

    # Gauss–Lobatto–Legendre (GLL) points
    def gll_points_weights(self, N: int) -> tuple[np.ndarray, np.ndarray]:
        """See https://mathworld.wolfram.com/LobattoQuadrature.html, but where our N is
        their N-1"""

        # One Legendre spectral element of order N.
        Pn = Legendre.basis(N)

        # Interior GLL points correspond to roots fo dPn/dx
        dPn = Pn.deriv()
        x_int = dPn.roots()

        # Add points at -1 and 1
        x = np.zeros(N + 1)
        x[0] = -1
        x[-1] = 1
        x[1:-1] = x_int

        # Set weights
        w = np.zeros(N + 1)
        w[0] = 2 / (N * (N + 1))
        w[-1] = 2 / (N * (N + 1))
        for i in range(1, N):
            w[i] = 2 / (N * (N + 1) * (Pn(x[i]) ** 2))

        return x, w

    def derivative_matrix(self, x: np.ndarray) -> np.ndarray:
        """See Deville+2002 Eq 2.4.9"""
        N = len(x) - 1
        D = np.zeros((N + 1, N + 1))

        # Evaluate Legendre polynomial L_N at GLL points
        Pn = Legendre.basis(N)
        L_N = Pn(x)

        for i in range(N + 1):
            for j in range(N + 1):
                if i != j:
                    D[i, j] = L_N[i] / L_N[j] / (x[i] - x[j])
                else:
                    # Diagonal elements
                    if i == 0:
                        D[i, i] = -N * (N + 1) / 4
                    elif i == N:
                        D[i, i] = N * (N + 1) / 4
                    else:
                        D[i, i] = 0

        return D

    def step(self) -> None:
        cfg = self.cfg

        # Compute derivatives
        Du = self.D @ self.u
        Db = self.D @ self.b

        # RHS for interior points
        f_u = self.M @ (
            self.u / cfg.dt - cfg.S * self.u * Du + cfg.S * self.b * Db
        )  # Fournier+2007 Eq. 26
        f_b = self.M @ (
            self.b / cfg.dt - cfg.S * self.u * Db + cfg.S * self.b * Du
        )  # Fournier+2007 Eq. 27

        # Enforce BCs
        f_u[0] = cfg.u_left
        f_u[-1] = cfg.u_right
        f_b[0] = cfg.b_left
        f_b[-1] = cfg.b_right

        # Solve (Fournier+2007 Eq. 23)
        self.u = np.linalg.solve(self.H_u, f_u)
        self.b = np.linalg.solve(self.H_b, f_b)

        # Reset BC exact values to avoid error accumulation
        self.u[0] = cfg.u_left
        self.u[-1] = cfg.u_right
        self.b[0] = cfg.b_left
        self.b[-1] = cfg.b_right

    def run(self) -> None:
        cfg = self.cfg
        steps = int(round(cfg.T / cfg.dt))
        snapshot_steps = {int(round(t / cfg.dt)) for t in (cfg.snapshot_times or [])}

        snapshot_num = 0
        for step in range(steps + 1):
            if step in snapshot_steps:
                self.write_snapshot(step * cfg.dt, snapshot_num)
                snapshot_num += 1
            if step < steps:
                self.step()

    def write_snapshot(self, t: float, snapshot_num: int) -> None:
        cfg = self.cfg
        path = Path(cfg.snapshot_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"snapshot_{snapshot_num:04d}.hdf5"

        with h5py.File(file_path, "w") as f:
            params = f.create_group("Parameters")
            params.attrs["S"] = cfg.S
            params.attrs["Pm"] = cfg.Pm
            params.attrs["N"] = cfg.N
            params.attrs["dt"] = cfg.dt
            params.attrs["T"] = cfg.T
            params.attrs["t"] = t

            grid = f.create_group("Coords")
            grid.create_dataset("x", data=self.x)

            fields = f.create_group("Fields")
            fields.create_dataset("u", data=self.u)
            fields.create_dataset("b", data=self.b)


if __name__ == "__main__":
    cfg = Config()
    solver = MHDSolver1D(cfg)
    solver.run()
