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
    from matplotlib.axes import Axes
    from matplotlib.typing import MarkerType

    from pinninit import Coords

    TensorDict = dict[str, torch.Tensor]


def load_data(
    filename: str,
    dataset_label: str,
    role_label: str,
    coords_class: type[Coords],
    device: torch.device,
) -> tuple[Coords, TensorDict]:
    with h5py.File(filename, "r") as f:
        ds_grp: h5py.Group = f[dataset_label][role_label]

        # Coordinates
        t = torch.tensor(
            ds_grp["Coords"]["t"][:],
            device=device,
        )

        # True Fourier coefficients, shape (n_points, 6)
        c_true = torch.tensor(
            ds_grp["Fields"]["c"][:],
            device=device,
        )

    coords = coords_class(t=t)
    fields_dict: TensorDict = {"c": c_true}

    return coords, fields_dict


# Scatter plotting
def plot_training_scatter(
    ax: Axes,
    i: int,
    coords: Coords,
    fields: TensorDict,
    label: str,
    marker: MarkerType,
    color: str,
) -> None:
    """The training points, at their values of coefficient i."""
    ax.scatter(
        coords.get_coord("t").cpu().numpy(),
        fields["c"][:, i].cpu().numpy(),
        s=25,
        marker=marker,
        color=color,
        edgecolors="k",
        linewidths=0.5,
        label=label,
        zorder=3,
    )


def plot_model_output(
    output_filepath: str | Path,
    test_coords: Coords,
    test_fields: TensorDict,
    test_model_output: torch.Tensor,
    data_loss: tuple[Coords, TensorDict],
    phys_loss: tuple[Coords, TensorDict],
) -> None:
    # Test data: the model outputs the 6 Fourier coefficients
    t = test_coords.get_coord("t").detach().cpu().numpy().squeeze()
    c_true = test_fields["c"].detach().cpu().numpy()
    c_pred = test_model_output.cpu().numpy()

    idx = np.argsort(t)
    t = t[idx]
    c_true = c_true[idx]
    c_pred = c_pred[idx]

    # One panel per coefficient: a_k on the left, b_k on the right
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes.flat):
        ax.plot(t, c_true[:, i], "k-", label="True")
        ax.plot(t, c_pred[:, i], "r--", label="PINN prediction")

        # Scatter points for training sets
        plot_training_scatter(
            ax, i, *phys_loss, "Training collocation points", "^", "green"
        )
        plot_training_scatter(ax, i, *data_loss, "Training data points", "o", "blue")

        ax.set_ylim((-1.5, 1.5))
        ax.set_ylabel(f"${'ab'[i % 2]}_{i // 2 + 1}(t)$")
    axes[0, 0].legend(loc="lower left", fontsize="small")
    for ax in axes[-1]:
        ax.set_xlabel(r"$t$")
    fig.tight_layout()

    # Save
    path = Path(output_filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    PARAM_FILE = "./param_travelling_wave_fourier_1D.yml"
    MODEL_PATH = "./models/travelling_wave_fourier_1D.pth"

    TRAIN_DATA_FILE = "./data/travelling_wave_fourier_1D_train_val.hdf5"
    TEST_DATA_FILE = "./data/travelling_wave_fourier_1D_test.hdf5"

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Coords class matching the training configuration
    coord_system = CartesianCoords(coord_axes=set(), device=device)
    coords_class = make_coords_dataclass({"t"}, coord_system, device)

    # Load test data
    test_coords, test_fields = load_data(
        TEST_DATA_FILE,
        "test",
        "test",
        coords_class,
        device,
    )

    # Load training data
    data_loss = load_data(
        TRAIN_DATA_FILE,
        "data_loss",
        "train",
        coords_class,
        device,
    )

    phys_loss = load_data(
        TRAIN_DATA_FILE,
        "phys_loss",
        "train",
        coords_class,
        device,
    )

    # Load PINN
    pinn = pinninit.PINN.from_yaml(PARAM_FILE, device)
    pinn.load(MODEL_PATH)

    # Evaluate model
    with torch.no_grad():
        test_model_output = pinn.evaluate_model(test_coords)

    output_filepath = "plots/travelling_wave_fourier_1D.png"

    # Plot
    plot_model_output(
        output_filepath,
        test_coords,
        test_fields,
        test_model_output,
        data_loss,
        phys_loss,
    )
