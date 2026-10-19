from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.stats import qmc
from sim_config import Config

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords

SEED = 1
VAL_FRACTION = 0.25  # Validation points per training point

N_X_DATA = 21  # Data points along x, training and validation together
N_T_DATA = 21  # Data points along t, training and validation together
N_BC = 128  # Training points of the boundary conditions at each end
N_PHYS = 8192  # Training collocation points of each equation


def snapshot_indices(
    points: np.ndarray, bounds: tuple[float, float], n: int
) -> np.ndarray:
    """Indices of n evenly spaced snapshot points (times or x) spanning bounds."""
    first, last = (int(np.argmin(np.abs(points - p))) for p in bounds)
    indices = np.linspace(first, last, n).round().astype(int)
    if not np.allclose(points[indices], np.linspace(*bounds, n)):
        raise ValueError(f"The snapshots don't have {n} even points in {bounds}.")
    return indices


def sobol_points(
    rng: np.random.Generator,
    n: int,
    x_bounds: tuple[float, float],
    t_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """n Sobol points. With a single x, the points are along t only."""
    single_x = x_bounds[0] == x_bounds[1]
    sobol = qmc.Sobol(d=1 if single_x else 2, scramble=True, seed=rng)
    unit = sobol.random_base2(int(np.ceil(np.log2(n))))[:n]
    if single_x:
        t = qmc.scale(unit, [t_bounds[0]], [t_bounds[1]])[:, 0]
        return {"x": np.full_like(t, x_bounds[0]), "t": t}
    points = qmc.scale(unit, [x_bounds[0], t_bounds[0]], [x_bounds[1], t_bounds[1]])
    return {"x": points[:, 0], "t": points[:, 1]}


if __name__ == "__main__":
    # Match the precision used in training
    torch.set_default_dtype(torch.float64)

    cfg = Config()
    with h5py.File(cfg.snapshot_path, "r") as f:
        times, x = f["t"][:], f["x"][:]
        u, b = f["u"][:], f["b"][:]

    x_bounds = (cfg.x_min, cfg.x_max)
    t_bounds = (0.0, cfg.T)
    rng = np.random.default_rng(SEED)
    coord_system = CartesianCoords(coord_axes={"x"}, device=torch.device("cpu"))
    datasets = pinninit.DataSetWriter(coord_system, {"x", "t"})

    # Data: b on a grid of snapshot points. Randomly chosen points are for
    # training, and the rest for validation
    idx_t = snapshot_indices(times, t_bounds, N_T_DATA)
    idx_x = snapshot_indices(x, x_bounds, N_X_DATA)
    t_grid, x_grid = np.meshgrid(times[idx_t], x[idx_x], indexing="ij")
    b_data = b[np.ix_(idx_t, idx_x)].ravel()
    n_data = b_data.size
    n_train = round(n_data / (1 + VAL_FRACTION))
    is_train = np.zeros(n_data, dtype=bool)
    is_train[rng.choice(n_data, n_train, replace=False)] = True
    for role, mask in (("train", is_train), ("val", ~is_train)):
        coords = {"x": x_grid.ravel()[mask], "t": t_grid.ravel()[mask]}
        datasets.add("data_loss", role, coords, {"b": b_data[mask]})

    # Boundary conditions: u and b at both ends, at the same times
    for role, n in (("train", N_BC), ("val", round(N_BC * VAL_FRACTION))):
        t = sobol_points(rng, n, (cfg.x_min, cfg.x_min), t_bounds)["t"]
        coords = {"x": np.repeat([cfg.x_min, cfg.x_max], n), "t": np.tile(t, 2)}
        fields = {
            "u": np.repeat([cfg.u_left, cfg.u_right], n),
            "b": np.repeat([cfg.b_left, cfg.b_right], n),
        }
        datasets.add("data_bc_loss", role, coords, fields)

    # Physics: the two equations over the whole domain. Validation points are
    # drawn separately
    for label in ("phys_momentum_loss", "phys_induction_loss"):
        n_val = round(N_PHYS * VAL_FRACTION)
        for role, n in (("train", N_PHYS), ("val", n_val)):
            datasets.add(label, role, sobol_points(rng, n, x_bounds, t_bounds), {})

    datasets.save("data/train_val.hdf5")

    # Test: u and b at every snapshot point
    t_grid, x_grid = np.meshgrid(times, x, indexing="ij")
    test = pinninit.DataSetWriter(coord_system, {"x", "t"})
    test.add(
        "test",
        "test",
        {"x": x_grid.ravel(), "t": t_grid.ravel()},
        {"u": u.ravel(), "b": b.ravel()},
    )
    test.save("data/test.hdf5")
