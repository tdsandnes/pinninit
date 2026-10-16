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
from pinninit.data_structure.coords import make_coords_dataclass
from pinninit.geometry.coordinate_system import CartesianCoords

if TYPE_CHECKING:
    from matplotlib.typing import MarkerType

    from pinninit.data_structure.coords import Coords

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
        x = torch.tensor(
            ds_grp["Coords"]["x"][:],
            device=device,
        )

        # Field
        f_true = torch.tensor(
            ds_grp["Fields"]["f"][:],
            device=device,
        )

    coords = coords_class(x=x)
    fields_dict: TensorDict = {"f": f_true}

    return coords, fields_dict


# Scatter plotting
def plot_training_scatter(
    pinn: pinninit.PINN,
    coords: Coords | None,
    label: str,
    marker: MarkerType,
    color: str,
) -> None:
    if coords is None:
        return

    with torch.no_grad():
        model_out = pinn.evaluate_model(coords)

    x_train = coords.get_coord("x").detach().cpu().numpy()
    f_train = model_out[:, 0].cpu().numpy()

    idx = np.argsort(x_train)
    x_train = x_train[idx]
    f_train = f_train[idx]

    plt.scatter(
        x_train,
        f_train,
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
    pinn: pinninit.PINN,
    test_coords: Coords,
    test_fields: TensorDict,
    test_model_output: torch.Tensor,
    data_loss_coords: Coords | None = None,
    phys_loss_coords: Coords | None = None,
) -> None:
    # Test data
    x = test_coords.get_coord("x").detach().cpu().numpy().squeeze()
    f_true = test_fields["f"].detach().cpu().numpy().squeeze()
    f_pred = test_model_output[:, 0].cpu().numpy()

    idx = np.argsort(x)
    x = x[idx]
    f_true = f_true[idx]
    f_pred = f_pred[idx]

    plt.figure(figsize=(6, 4))
    plt.plot(x, f_true, "k-", label="f(x)")
    plt.plot(x, f_pred, "r--", label="PINN prediction")

    # Scatter points for training sets
    plot_training_scatter(
        pinn,
        phys_loss_coords,
        "Training collocation points",
        "^",
        "green",
    )
    plot_training_scatter(
        pinn,
        data_loss_coords,
        "Training data points",
        "o",
        "blue",
    )

    plt.ylim((-1.5, 3.5))
    plt.xlabel(r"$x$")
    plt.ylabel(r"$f(x)$")
    plt.legend()
    plt.tight_layout()

    # Save
    path = Path(output_filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    PARAM_FILE = "./param_simple_harmonic_oscillator_1D.yml"
    MODEL_PATH = "./models/simple_harmonic_oscillator_1D.pth"

    TRAIN_DATA_FILE = "./data/simple_harmonic_oscillator_1D_train_val.hdf5"
    TEST_DATA_FILE = "./data/simple_harmonic_oscillator_1D_test.hdf5"

    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Coords class matching the training configuration
    coord_system = CartesianCoords(dim_geometric=1, coord_axes={"x"}, device=device)
    coords_class = make_coords_dataclass({"x"}, coord_system, device)

    # Load test data
    test_coords, test_fields = load_data(
        TEST_DATA_FILE,
        "test",
        "test",
        coords_class,
        device,
    )

    # Load training data
    data_loss_coords, _ = load_data(
        TRAIN_DATA_FILE,
        "data_loss",
        "train",
        coords_class,
        device,
    )

    phys_loss_coords, _ = load_data(
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

    output_filepath = "plots/simple_harmonic_oscillator_1D.png"

    # Plot
    plot_model_output(
        output_filepath,
        pinn,
        test_coords,
        test_fields,
        test_model_output,
        data_loss_coords,
        phys_loss_coords,
    )
