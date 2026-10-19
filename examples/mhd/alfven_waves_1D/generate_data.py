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
T_BOUNDS = (20.0, 30.0)  # Training time window
T_BOUNDS_TEST = (20.0, 30.0)  # Test time window
VAL_FRACTION = 0.25  # Validation points per training point

# NOTE: The paper's fiducial setup uses N_DATA = 1025, N_BC = 16384 and
# N_PHYS = 32768. Fewer points are used here for the example.
N_DATA = 257  # Data points at x = L, training and validation together
N_BC = 1024  # Training points of the db/dx = 0 condition at x = L
N_PHYS = 4096  # Training collocation points of each equation
N_X_TEST = 257
N_T_TEST = 257


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

    rng = np.random.default_rng(SEED)
    coord_system = CartesianCoords(coord_axes={"x"}, device=torch.device("cpu"))
    datasets = pinninit.DataSetWriter(coord_system, {"x", "t"})

    # Data: b at x = L at evenly spaced times. Randomly chosen points are for
    # training, and the rest for validation
    idx_t = snapshot_indices(times, T_BOUNDS, N_DATA)
    t_data, b_data = times[idx_t], b[idx_t, -1]
    n_train = round(N_DATA / (1 + VAL_FRACTION))
    is_train = np.zeros(N_DATA, dtype=bool)
    is_train[rng.choice(N_DATA, n_train, replace=False)] = True
    for role, mask in (("train", is_train), ("val", ~is_train)):
        coords = {"x": np.full(mask.sum(), cfg.L), "t": t_data[mask]}
        datasets.add("data_loss", role, coords, {"b": b_data[mask]})

    # Physics: the boundary condition at x = L, and the two equations over
    # the whole domain. Validation points are drawn separately
    for label, n, x_bounds in (
        ("dbdx_bc_loss", N_BC, (cfg.L, cfg.L)),
        ("phys_zp_loss", N_PHYS, (0.0, cfg.L)),
        ("phys_zm_loss", N_PHYS, (0.0, cfg.L)),
    ):
        n_val = round(n * VAL_FRACTION)
        for role, n_role in (("train", n), ("val", n_val)):
            datasets.add(label, role, sobol_points(rng, n_role, x_bounds, T_BOUNDS), {})

    datasets.save("data/train_val.hdf5")

    # Test: u and b on a grid of snapshot points over the test window
    idx_t = snapshot_indices(times, T_BOUNDS_TEST, N_T_TEST)
    idx_x = snapshot_indices(x, (0.0, cfg.L), N_X_TEST)
    t_grid, x_grid = np.meshgrid(times[idx_t], x[idx_x], indexing="ij")
    test = pinninit.DataSetWriter(coord_system, {"x", "t"})
    test.add(
        "test",
        "test",
        {"x": x_grid.ravel(), "t": t_grid.ravel()},
        {
            "u": u[np.ix_(idx_t, idx_x)].ravel(),
            "b": b[np.ix_(idx_t, idx_x)].ravel(),
        },
    )
    test.save("data/test.hdf5")
