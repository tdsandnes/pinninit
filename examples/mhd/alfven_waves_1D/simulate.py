from __future__ import annotations

from pathlib import Path
from typing import Any

import dedalus.public as d3
import h5py
import numpy as np
from scipy.interpolate import BarycentricInterpolator
from sim_config import Config


def simulate(cfg: Config) -> None:
    # Domain
    xcoord = d3.Coordinate("x")
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.Chebyshev(xcoord, size=cfg.N, bounds=(0, cfg.L), dealias=cfg.dealias)

    # Fields and tau terms
    zp = dist.Field(name="zp", bases=xbasis)
    zm = dist.Field(name="zm", bases=xbasis)
    zm_in = dist.Field(name="zm_in")
    tau_zp1 = dist.Field(name="tau_zp1")
    tau_zp2 = dist.Field(name="tau_zp2")
    tau_zm1 = dist.Field(name="tau_zm1")
    tau_zm2 = dist.Field(name="tau_zm2")

    # Parameters, used by the equations below
    Ma = cfg.Ma
    nu_p = Ma * 0.5 * (1 / cfg.Re + 1 / cfg.Rm)
    nu_m = Ma * 0.5 * (1 / cfg.Re - 1 / cfg.Rm)
    L = cfg.L

    def lift(A: Any, n: int) -> Any:
        return d3.Lift(A, xbasis, n)

    def dx(A: Any) -> Any:
        return d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP(
        [zp, zm, tau_zp1, tau_zp2, tau_zm1, tau_zm2],
        namespace=locals(),
    )
    problem.add_equation(
        "dt(zp) - dx(zp) - nu_p*dx(dx(zp)) - nu_m*dx(dx(zm))"
        " + lift(tau_zp1, -1) + lift(tau_zp2, -2) = -Ma*zm*dx(zp)"
    )
    problem.add_equation(
        "dt(zm) + dx(zm) - nu_p*dx(dx(zm)) - nu_m*dx(dx(zp))"
        " + lift(tau_zm1, -1) + lift(tau_zm2, -2) = -Ma*zp*dx(zm)"
    )

    # Forcing at x = 0
    problem.add_equation("zm(x=0) = zm_in")
    problem.add_equation("dx(zp)(x=0) = 0")

    # Reflection at x = L
    problem.add_equation("zp(x=L) + zm(x=L) = 0")
    problem.add_equation("dx(zp)(x=L) - dx(zm)(x=L) = 0")

    solver = problem.build_solver(d3.SBDF2)

    # The fields start at zero. To save snapshots on a uniform grid, the
    # Chebyshev polynomial of each field is evaluated from its grid values
    x_grid = dist.local_grid(xbasis, scale=1)
    x_out = np.linspace(0, cfg.L, cfg.N_x_out)
    to_x_out = BarycentricInterpolator(x_grid, np.eye(cfg.N))(x_out)

    def on_x_out(field: Any) -> np.ndarray:
        field.change_scales(1)
        values: np.ndarray = to_x_out @ field["g"]
        return values

    # Step, saving snapshots before each step
    n_steps = round(cfg.T / cfg.dt)
    first_snapshot = round(cfg.t_snapshots_start / cfg.dt)
    times, u, b = [], [], []
    for step in range(n_steps + 1):
        if step >= first_snapshot and (step - first_snapshot) % cfg.snapshot_steps == 0:
            zp_out, zm_out = on_x_out(zp), on_x_out(zm)
            times.append(step * cfg.dt)
            u.append(0.5 * (zp_out + zm_out))
            b.append(0.5 * (zp_out - zm_out))

        if step < n_steps:
            zm_in["g"] = cfg.zm_in(solver.sim_time)
            solver.step(cfg.dt)
            if solver.iteration % 1024 == 0:
                print(f"t = {solver.sim_time:.3f}")

    # Save snapshots
    path = Path(cfg.snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=np.array(times))
        f.create_dataset("x", data=x_out)
        f.create_dataset("u", data=np.array(u))
        f.create_dataset("b", data=np.array(b))
    print(f"Saved {len(times)} snapshots to {path}")


if __name__ == "__main__":
    simulate(Config())
