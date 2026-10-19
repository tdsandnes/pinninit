from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords

### Generates and saves datasets for the simple harmonic oscillator example.


def compute_f(x: float | np.ndarray) -> float | np.ndarray:
    return np.sin(x)


def generate_datasets(
    label: str,
    datasets: pinninit.DataSetWriter,
    x_bounds: tuple[float, float],
    alpha: float,
    roles: list[str],
    seeds: dict[str, int],
    with_shift: dict[str, bool],
    with_fields: bool = True,
    with_delta: bool = False,
    delta: float | None = None,
) -> None:
    """Generate and add datasets for all roles."""
    x_min, x_max = x_bounds
    period = 2 * np.pi
    N = int(alpha * (x_max - x_min) / period)

    for role in roles:
        # Generate evenly spaced coordinates
        x = np.linspace(x_min, x_max, N, endpoint=False)

        if with_shift[role]:
            # Use the provided seed for reproducibility
            rng = np.random.default_rng(seeds[role])

            # Apply a random shift within the x bounds
            shift = rng.uniform(0, x_max - x_min)
            x = (x + shift - x_min) % (x_max - x_min) + x_min

        x = np.sort(x)
        if with_delta:
            if delta is None:
                raise ValueError("with_delta used without giving delta.")
            x -= delta * np.arange(len(x))

        coords = {"x": x}

        fields = {"f": compute_f(x)} if with_fields else None

        datasets.add(label=label, role=role, coords=coords, fields=fields)


if __name__ == "__main__":
    # Match the precision used in training
    torch.set_default_dtype(torch.float64)

    # Params
    n_wavelengths_data = 2
    n_wavelengths_col = 4
    x_bounds_data = (0, 2 * np.pi * n_wavelengths_data)
    x_bounds_col = (0, 2 * np.pi * n_wavelengths_col)
    alpha_data = 2
    alpha_col = 10

    # Init dataset writer
    coord_system = CartesianCoords(coord_axes={"x"}, device=torch.device("cpu"))
    datasets = pinninit.DataSetWriter(coord_system, {"x"})

    # Define roles and rng for seeds
    roles = ["train", "val"]
    seed = 12345
    rng = np.random.default_rng(seed)

    # Set shift delta
    delta = 1e-2 * 2 * np.pi

    # Generate data loss points
    generate_datasets(
        label="data_loss",
        datasets=datasets,
        x_bounds=x_bounds_data,
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
        x_bounds=x_bounds_col,
        alpha=alpha_col,
        roles=roles,
        seeds={role: int(rng.integers(0, 1000000)) for role in roles},
        with_shift={"train": True, "val": True},
        with_fields=True,
        with_delta=False,
    )

    # Save HDF5
    datasets.save("data/simple_harmonic_oscillator_1D_train_val.hdf5")

    # Now generate test data
    alpha_test = 100

    # Data writer for test set
    datasets_test = pinninit.DataSetWriter(coord_system, {"x"})

    # Define roles
    roles_test = ["test"]

    n_wavelengths_test = 6
    x_bounds_test = (0, 2 * np.pi * n_wavelengths_test)

    # Generate test points
    generate_datasets(
        label="test",
        datasets=datasets_test,
        x_bounds=x_bounds_test,
        alpha=alpha_test,
        roles=roles_test,
        seeds={role: int(rng.integers(0, 1000000)) for role in roles_test},
        with_shift={"test": False},
        with_fields=True,
        with_delta=False,
    )

    # Save HDF5
    datasets_test.save("data/simple_harmonic_oscillator_1D_test.hdf5")
