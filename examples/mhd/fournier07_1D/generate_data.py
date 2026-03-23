from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from sim_config import Config

sys.path.append(str(Path(__file__).resolve().parents[2]))
from datasets import DataSets

### Super WIP


class SnapshotLoader:
    def __init__(self, snapshot_dir: str) -> None:
        self.paths = sorted(Path(snapshot_dir).glob("snapshot_*.hdf5"))
        if not self.paths:
            raise ValueError(f"No snapshots found in {snapshot_dir}")

        times: list[float] = []
        for p in self.paths:
            with h5py.File(p, "r") as f:
                times.append(float(f["Parameters"].attrs["t"]))

        self.times = np.array(times)

    def get_snapshot_index(self, t: float) -> int:
        mask = np.isclose(self.times, t)
        n_matches = np.count_nonzero(mask)

        if n_matches == 0:
            raise ValueError(f"Requested time {t} does not match any snapshot.")
        if n_matches > 1:
            raise ValueError(f"Requested time {t} matches multiple snapshots.")

        return int(np.flatnonzero(mask)[0])

    def load_snapshot(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self.paths[idx]
        with h5py.File(path, "r") as f:
            x = f["Coords"]["x"][:]
            u = f["Fields"]["u"][:]
            b = f["Fields"]["b"][:]
        return x, u, b


def sample_snapshot(
    loader: SnapshotLoader, x_value: np.ndarray, idx_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample u(x,t), b(x,t) at snapshot idx_t.
    """
    U = np.zeros((len(idx_t), len(x_value)))
    B = np.zeros_like(U)

    for i, idx in enumerate(idx_t):
        x, u, b = loader.load_snapshot(idx)
        U[i] = np.interp(x_value, x, u)
        B[i] = np.interp(x_value, x, b)

    return U, B


def generate_mhd_dataset(
    loader: SnapshotLoader,
    datasets: DataSets,
    label: str,
    role: str,
    x_bounds: tuple[float, float],
    full_t_values: np.ndarray,
    Nt: int,
    Nx: int,
    seed: int | None = None,
    shift_x: bool = True,
    shift_t: bool = True,
    include_u: bool = True,
    t_bounds: tuple[float, float] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    x_min, x_max = x_bounds
    N_snapshots = len(full_t_values)

    dx = (x_max - x_min) / Nx
    x = x_min + dx * np.arange(Nx)
    if shift_x:
        shift_x_val = rng.uniform(0, dx)
        x = (x + shift_x_val - x_min) % (x_max - x_min) + x_min

    dt_idx = N_snapshots // Nt
    idx_t = np.arange(Nt) * dt_idx

    use_snapshots = label in ("data_loss", "test")

    if use_snapshots:
        if shift_t:
            dt_idx = N_snapshots // Nt
            idx_t = np.arange(Nt) * dt_idx
            shift_idx = rng.integers(0, max(1, dt_idx))
            idx_t = (idx_t + shift_idx) % N_snapshots
        else:
            idx_t = np.round(np.linspace(0, N_snapshots - 1, Nt)).astype(int)

        t_values = full_t_values[idx_t]
        U, B = sample_snapshot(loader, x, idx_t)
    else:
        if t_bounds is None:
            raise ValueError("t_bounds must be provided for non-data datasets")

        t_min, t_max = t_bounds

        if shift_t:
            dt = (t_max - t_min) / Nt
            t_values = t_min + dt * np.arange(Nt)
            shift = rng.uniform(0, dt)
            t_values = t_values + shift
        else:
            t_values = np.linspace(t_min, t_max, Nt)

        U = B = None

    T_grid, X_grid = np.meshgrid(t_values, x, indexing="ij")
    coords = {"t": T_grid.ravel(), "x": X_grid.ravel()}

    fields = {}
    if use_snapshots:
        fields["b"] = B.ravel()
        if include_u:
            fields["u"] = U.ravel()

    datasets.add_dataset(label=label, dataset_type=role, coords=coords, fields=fields)


def generate_BC_dataset(
    datasets: DataSets,
    label: str,
    role: str,
    x_bounds: tuple[float, float],
    full_t_values: np.ndarray,
    Nt: int,
    u_left: float,
    u_right: float,
    b_left: float,
    b_right: float,
    seed: int | None = None,
    shift_t: bool = True,
    t_bounds: tuple[float, float] | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    x_min, x_max = x_bounds
    N_snapshots = len(full_t_values)

    x = np.array([x_min, x_max])

    if t_bounds is None:
        dt_idx = max(1, N_snapshots // Nt)
        idx_t = np.arange(Nt) * dt_idx
        if shift_t:
            shift_idx = rng.integers(0, dt_idx)
            idx_t = (idx_t + shift_idx) % N_snapshots
        t_values = full_t_values[idx_t]
    else:
        t_min, t_max = t_bounds
        if shift_t:
            dt = (t_max - t_min) / Nt
            t_values = t_min + dt * np.arange(Nt)
            shift = rng.uniform(0, dt)
            t_values = t_values + shift
        else:
            t_values = np.linspace(t_min, t_max, Nt)

    U = np.tile(np.array([u_left, u_right]), (Nt, 1))
    B = np.tile(np.array([b_left, b_right]), (Nt, 1))

    T_grid, X_grid = np.meshgrid(t_values, x, indexing="ij")
    coords = {"t": T_grid.ravel(), "x": X_grid.ravel()}

    fields = {"u": U.ravel(), "b": B.ravel()}

    datasets.add_dataset(label=label, dataset_type=role, coords=coords, fields=fields)


if __name__ == "__main__":
    cfg = Config()
    loader = SnapshotLoader(cfg.snapshot_dir)

    full_t_values = np.array(cfg.snapshot_times)
    x_bounds = (cfg.x_min, cfg.x_max)

    datasets_train_val = DataSets()

    Nx_data_train, Nt_data_train = 20, 20
    Nx_data_val, Nt_data_val = 20, 20

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="data_loss",
        role="training",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_data_train,
        Nx=Nx_data_train,
        seed=1,
        shift_x=True,
        shift_t=False,
        include_u=True,
    )

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="data_loss",
        role="validation",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_data_val,
        Nx=Nx_data_val,
        seed=2,
        shift_x=True,
        shift_t=True,
        include_u=True,
    )

    t_bounds = (0.0, 0.2)  # (-0.05, 0.25)

    Nt_BC_train = 100  # 50
    Nt_BC_val = 100
    u_left = 0.0
    u_right = 0.0
    b_left = -1.0
    b_right = 1.0

    generate_BC_dataset(
        datasets_train_val,
        label="data_bc_loss",
        role="training",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_BC_train,
        u_left=u_left,
        u_right=u_right,
        b_left=b_left,
        b_right=b_right,
        seed=3,
        shift_t=False,
        t_bounds=t_bounds,
    )

    generate_BC_dataset(
        datasets_train_val,
        label="data_bc_loss",
        role="validation",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_BC_val,
        u_left=u_left,
        u_right=u_right,
        b_left=b_left,
        b_right=b_right,
        seed=4,
        shift_t=True,
        t_bounds=t_bounds,
    )

    Nx_phys_momentum_train, Nt_phys_momentum_train = 100, 100
    Nx_phys_momentum_val, Nt_phys_momentum_val = 100, 100

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="phys_momentum_loss",
        role="training",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_phys_momentum_train,
        Nx=Nx_phys_momentum_train,
        seed=5,
        shift_x=True,
        shift_t=True,
        include_u=True,
        t_bounds=t_bounds,
    )

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="phys_momentum_loss",
        role="validation",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_phys_momentum_val,
        Nx=Nx_phys_momentum_val,
        seed=6,
        shift_x=True,
        shift_t=True,
        include_u=True,
        t_bounds=t_bounds,
    )

    Nx_phys_induction_train, Nt_phys_induction_train = 100, 100
    Nx_phys_induction_val, Nt_phys_induction_val = 100, 100

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="phys_induction_loss",
        role="training",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_phys_induction_train,
        Nx=Nx_phys_induction_train,
        seed=7,
        shift_x=True,
        shift_t=True,
        include_u=True,
        t_bounds=t_bounds,
    )

    generate_mhd_dataset(
        loader,
        datasets_train_val,
        label="phys_induction_loss",
        role="validation",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_phys_induction_val,
        Nx=Nx_phys_induction_val,
        seed=8,
        shift_x=True,
        shift_t=True,
        include_u=True,
        t_bounds=t_bounds,
    )

    output_path_train_val = "data/fournier07_1D_train_val.hdf5"
    datasets_train_val.save(output_path_train_val)

    datasets_test = DataSets()
    Nx_test, Nt_test = 300, len(full_t_values)
    generate_mhd_dataset(
        loader,
        datasets_test,
        label="test",
        role="test",
        x_bounds=x_bounds,
        full_t_values=full_t_values,
        Nt=Nt_test,
        Nx=Nx_test,
        seed=9,
        shift_x=True,
        shift_t=False,
        include_u=True,
    )

    output_path_test = "data/fournier07_1D_test.hdf5"
    datasets_test.save(output_path_test)
