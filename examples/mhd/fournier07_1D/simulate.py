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
    xbasis = d3.Chebyshev(
        xcoord, size=cfg.N, bounds=(cfg.x_min, cfg.x_max), dealias=cfg.dealias
    )

    # Fields and tau terms
    u = dist.Field(name="u", bases=xbasis)
    b = dist.Field(name="b", bases=xbasis)
    tau_u1 = dist.Field(name="tau_u1")
    tau_u2 = dist.Field(name="tau_u2")
    tau_b1 = dist.Field(name="tau_b1")
    tau_b2 = dist.Field(name="tau_b2")

    # Parameters, used by the equations below
    S, Pm = cfg.S, cfg.Pm
    x_min, x_max = cfg.x_min, cfg.x_max
    u_left, u_right = cfg.u_left, cfg.u_right
    b_left, b_right = cfg.b_left, cfg.b_right

    def lift(A: Any, n: int) -> Any:
        return d3.Lift(A, xbasis, n)

    def dx(A: Any) -> Any:
        return d3.Differentiate(A, xcoord)

    # Problem
    problem = d3.IVP([u, b, tau_u1, tau_u2, tau_b1, tau_b2], namespace=locals())
    problem.add_equation(
        "dt(u) - Pm*dx(dx(u)) + lift(tau_u1, -1) + lift(tau_u2, -2)"
        " = -S*u*dx(u) + S*b*dx(b)"
    )
    problem.add_equation(
        "dt(b) - dx(dx(b)) + lift(tau_b1, -1) + lift(tau_b2, -2)"
        " = -S*u*dx(b) + S*b*dx(u)"
    )
    problem.add_equation("u(x=x_min) = u_left")
    problem.add_equation("u(x=x_max) = u_right")
    problem.add_equation("b(x=x_min) = b_left")
    problem.add_equation("b(x=x_max) = b_right")

    solver = problem.build_solver(d3.SBDF2)

    # Initial conditions
    x_grid = dist.local_grid(xbasis, scale=1)
    u["g"] = cfg.u_initial(x_grid)
    b["g"] = cfg.b_initial(x_grid)

    # To save snapshots on a uniform grid, the Chebyshev polynomial of each
    # field is evaluated from its grid values
    x_out = np.linspace(cfg.x_min, cfg.x_max, cfg.N_x_out)
    to_x_out = BarycentricInterpolator(x_grid, np.eye(cfg.N))(x_out)

    def on_x_out(field: Any) -> np.ndarray:
        field.change_scales(1)
        values: np.ndarray = to_x_out @ field["g"]
        return values

    # Step, saving snapshots before each step
    n_steps = round(cfg.T / cfg.dt)
    times, u_out, b_out = [], [], []
    for step in range(n_steps + 1):
        if step % cfg.snapshot_steps == 0:
            times.append(step * cfg.dt)
            u_out.append(on_x_out(u))
            b_out.append(on_x_out(b))
        if step < n_steps:
            solver.step(cfg.dt)

    # Save snapshots
    path = Path(cfg.snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("t", data=np.array(times))
        f.create_dataset("x", data=x_out)
        f.create_dataset("u", data=np.array(u_out))
        f.create_dataset("b", data=np.array(b_out))
    print(f"Saved {len(times)} snapshots to {path}")


if __name__ == "__main__":
    simulate(Config())
