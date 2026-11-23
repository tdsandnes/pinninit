from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords

# Positions x of the observatories, evenly spaced around the periodic domain
OBSERVATORIES = np.linspace(0, 2 * np.pi, 8, endpoint=False)

# Wavenumber, amplitude and phase of each Fourier coefficient
# (a_1, b_1, a_2, b_2, a_3, b_3)
WAVENUMBERS = np.array([1, 1, 2, 2, 3, 3])
AMPLITUDES = np.array([1.0, 0.7, 0.9, 1.2, 0.8, 0.6])
PHASES = np.array([0.0, 1.0, 2.0, 0.5, 1.5, 2.5])


def compute_c(t: np.ndarray) -> np.ndarray:
    """The Fourier coefficients at time t, shape (n_points, 6). Each is a
    simple harmonic oscillator, c_j = R_j cos(k_j t + phi_j), which is the
    general solution of the wave equation for its mode."""
    c: np.ndarray = AMPLITUDES * np.cos(np.outer(t, WAVENUMBERS) + PHASES)
    return c


def compute_f(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The field at time t and position x,
    f = sum_k a_k cos(k x) + b_k sin(k x)."""
    kx = np.outer(x, WAVENUMBERS)
    basis = np.where(np.arange(6) % 2 == 0, np.cos(kx), np.sin(kx))
    f: np.ndarray = (compute_c(t) * basis).sum(axis=1)
    return f


def generate_datasets(
    label: str,
    datasets: pinninit.DataSetWriter,
    t_bounds: tuple[float, float],
    alpha: float,
    roles: list[str],
    seeds: dict[str, int],
    with_shift: dict[str, bool],
    with_fields: bool = True,
    with_delta: bool = False,
    delta: float | None = None,
) -> None:
    """Generate and add datasets for all roles."""
    t_min, t_max = t_bounds
    period = 2 * np.pi
    N = int(alpha * (t_max - t_min) / period)

    for role in roles:
        # Generate evenly spaced times
        t = np.linspace(t_min, t_max, N, endpoint=False)

        if with_shift[role]:
            # Use the provided seed for reproducibility
            rng = np.random.default_rng(seeds[role])

            # Apply a random shift within the t bounds
            shift = rng.uniform(0, t_max - t_min)
            t = (t + shift - t_min) % (t_max - t_min) + t_min

        t = np.sort(t)
        if with_delta:
            if delta is None:
                raise ValueError("with_delta used without giving delta.")
            t -= delta * np.arange(len(t))

        if with_fields:
            # Every observatory measures the field at each time. The positions
            # x are a known field, and c (not used in training) is for plotting
            x = np.tile(OBSERVATORIES, len(t))
            t = np.repeat(t, len(OBSERVATORIES))
            fields = {"x": x, "f": compute_f(t, x), "c": compute_c(t)}
        else:
            fields = {"c": compute_c(t)}

        coords = {"t": t}

        datasets.add(label=label, role=role, coords=coords, fields=fields)


if __name__ == "__main__":
    # Params
    n_wavelengths_data = 2
    n_wavelengths_col = 4
    t_bounds_data = (0, 2 * np.pi * n_wavelengths_data)
    t_bounds_col = (0, 2 * np.pi * n_wavelengths_col)
    alpha_data = 2
    alpha_col = 50

    # Init dataset writer
    # Time t is the only coordinate, so there are no geometric axes
    coord_system = CartesianCoords(coord_axes=set(), device=torch.device("cpu"))
    datasets = pinninit.DataSetWriter(coord_system, {"t"})

    # Define roles and rng for seeds
    roles = ["train", "val"]
    seed = 12345
    rng = np.random.default_rng(seed)

    # Set shift delta
    delta = 5e-3 * 2 * np.pi

    # Generate data loss points
    generate_datasets(
        label="data_loss",
        datasets=datasets,
        t_bounds=t_bounds_data,
        alpha=alpha_data,
        roles=roles,
        seeds={role: int(rng.integers(0, 1000000)) for role in roles},
        with_shift={"train": False, "val": True},
        with_fields=True,
        with_delta=True,
        delta=delta,
    )

    # Generate phys loss points
    generate_datasets(
        label="phys_loss",
        datasets=datasets,
        t_bounds=t_bounds_col,
        alpha=alpha_col,
        roles=roles,
        seeds={role: int(rng.integers(0, 1000000)) for role in roles},
        with_shift={"train": True, "val": True},
        with_fields=False,
        with_delta=False,
    )

    # Save HDF5
    datasets.save("data/travelling_wave_fourier_1D_train_val.hdf5")

    # Now generate test data
    alpha_test = 100

    # Data writer for test set
    datasets_test = pinninit.DataSetWriter(coord_system, {"t"})

    # Define roles
    roles_test = ["test"]

    n_wavelengths_test = 6
    t_bounds_test = (0, 2 * np.pi * n_wavelengths_test)

    # Generate test points
    generate_datasets(
        label="test",
        datasets=datasets_test,
        t_bounds=t_bounds_test,
        alpha=alpha_test,
        roles=roles_test,
        seeds={role: int(rng.integers(0, 1000000)) for role in roles_test},
        with_shift={"test": False},
        with_fields=False,
        with_delta=False,
    )

    # Save HDF5
    datasets_test.save("data/travelling_wave_fourier_1D_test.hdf5")
