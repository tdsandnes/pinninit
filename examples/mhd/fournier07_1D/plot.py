from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from typing import TYPE_CHECKING

import pinninit
from pinninit import CartesianCoords, make_coords_dataclass

if TYPE_CHECKING:
    from pinninit import Coords


def read_snapshot(path: Path) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Read time, x, u, b from an HDF5 snapshot."""
    with h5py.File(path, "r") as f:
        t = f["Parameters"].attrs["t"]
        x = f["Coords"]["x"][:]
        u = f["Fields"]["u"][:]
        b = f["Fields"]["b"][:]
    return t, x, u, b


def evaluate_pinn_at_test_points(
    pinn: pinninit.PINN,
    coords_class: type[Coords],
    x_test: np.ndarray,
    t_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate PINN at test coordinates for u and b."""
    coords = coords_class(
        x=torch.tensor(x_test, device=device),
        t=torch.tensor(t_test, device=device),
    )
    with torch.no_grad():
        output = pinn.evaluate_model(coords)

    # Select output columns by field label
    b_idx = pinn.model.output_field_labels.index("b")
    u_idx = pinn.model.output_field_labels.index("u")
    b_pred = output[:, b_idx].cpu().numpy()
    u_pred = output[:, u_idx].cpu().numpy()
    return u_pred, b_pred


def load_test_coords(
    test_file: Path, dataset_label: str = "test", role_label: str = "test"
) -> tuple[np.ndarray, np.ndarray]:
    """Load test coordinates from HDF5."""
    with h5py.File(test_file, "r") as f:
        x_test = f[dataset_label][role_label]["Coords"]["x"][:]
        t_test = f[dataset_label][role_label]["Coords"]["t"][:]
    return x_test, t_test


if __name__ == "__main__":
    PARAM_FILE = "./param_fournier07_1D.yml"
    MODEL_PATH = "./models/fournier07_1D.pth"
    TEST_DATA_FILE = Path("./data/fournier07_1D_test.hdf5")
    SNAPSHOT_DIR = Path("./data/snapshots")

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Coords class matching the training configuration
    coord_system = CartesianCoords(coord_axes={"x"}, device=device)
    coords_class = make_coords_dataclass({"x", "t"}, coord_system, device)

    # Load PINN
    pinn = pinninit.PINN.from_yaml(PARAM_FILE, device)
    pinn.load(MODEL_PATH)

    # Desired snapshot times
    desired_times = [0.0, 0.1, 0.2]

    # Load snapshot files and times
    snapshot_files = sorted(SNAPSHOT_DIR.glob("snapshot_*.hdf5"))
    snapshots = [read_snapshot(f) for f in snapshot_files]
    snapshot_times = np.array([snap[0] for snap in snapshots])

    # Verify desired times exist in snapshots
    snapshots_to_plot = []
    for t_desired in desired_times:
        if not np.any(np.isclose(snapshot_times, t_desired, atol=1e-12)):
            raise ValueError(
                f"Snapshot at time {t_desired} not found in snapshot files."
            )
        idx = np.where(np.isclose(snapshot_times, t_desired, atol=1e-12))[0][0]
        snapshots_to_plot.append(snapshots[idx])

    # Load test coordinates
    x_test_all, t_test_all = load_test_coords(TEST_DATA_FILE)

    # Verify desired times exist in test data
    for t_desired in desired_times:
        if not np.any(np.isclose(t_test_all, t_desired, atol=1e-12)):
            raise ValueError(f"Test data at time {t_desired} not found.")

    # Create 3 rows of 2 panels (u and b)
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)

    for i, (t_snap, x_snap, u_snap, b_snap) in enumerate(snapshots_to_plot):
        # Select test points corresponding to this snapshot time
        time_mask = np.isclose(t_test_all, t_snap, atol=1e-12)
        x_test = x_test_all[time_mask]
        t_test = t_test_all[time_mask]

        # Evaluate PINN at test points
        u_pred_test, b_pred_test = evaluate_pinn_at_test_points(
            pinn, coords_class, x_test, t_test, device
        )

        # Plot true solution from snapshot
        ax_u = axes[i, 0]
        ax_u.plot(x_snap, u_snap, label=f"Snapshot t={t_snap:.3f}")
        ax_u.plot(x_test, u_pred_test, "--", label="PINN (test points)")
        ax_u.set_ylabel("u")
        ax_u.legend()
        if i == 2:
            ax_u.set_xlabel("x")

        ax_b = axes[i, 1]
        ax_b.plot(x_snap, b_snap, label=f"Snapshot t={t_snap:.3f}")
        ax_b.plot(x_test, b_pred_test, "--", label="PINN (test points)")
        ax_b.set_ylabel("b")
        ax_b.legend()
        if i == 2:
            ax_b.set_xlabel("x")

    plt.tight_layout()
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / "fournier07_pinn.png", dpi=300)
    plt.show()
