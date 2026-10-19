from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from alfven_waves_1D_equations import u_eqn

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords, make_coords_dataclass


def load_test_data(path: str) -> dict[str, np.ndarray]:
    """Test coords and fields, as (t, x) grids."""
    with h5py.File(path, "r") as f:
        group = f["test"]["test"]
        data = {name: group["Coords"][name][:] for name in ("x", "t")}
        data.update({name: group["Fields"][name][:] for name in ("u", "b")})
    n_x = len(np.unique(data["x"]))
    return {name: values.reshape(-1, n_x) for name, values in data.items()}


def evaluate_pinn(
    pinn: pinninit.PINN, x: np.ndarray, t: np.ndarray, n_batches: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """The PINN's u and b at the points (x, t)."""
    device = pinn.device
    coord_system = CartesianCoords(coord_axes={"x"}, device=device)
    XTCoords = make_coords_dataclass({"x", "t"}, coord_system, device)
    labels = pinn.model.output_field_labels

    u, b = [], []
    for x_batch, t_batch in zip(
        np.array_split(x, n_batches), np.array_split(t, n_batches), strict=True
    ):
        coords = XTCoords(
            x=torch.tensor(x_batch, device=device),
            t=torch.tensor(t_batch, device=device),
        )
        out = pinn.evaluate_model(coords)
        nn_out = out[:, labels.index("nn_out")]
        x_coord = coords.get_coord("x")
        u.append(u_eqn(nn_out, x_coord, pinninit.Operators(coords)).cpu().numpy())
        b.append(out[:, labels.index("b")].cpu().numpy())
    return np.concatenate(u), np.concatenate(b)


def relative_error(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def plot_fields(test: dict[str, np.ndarray], pred: dict[str, np.ndarray]) -> None:
    """Simulation, PINN and their difference for u and b over (x, t)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    for row, name in zip(axes, ("u", "b"), strict=True):
        vmax = np.abs(test[name]).max()
        field_style = {"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax}
        error = np.abs(pred[name] - test[name])
        panels = (
            (f"Simulation {name}", test[name], field_style),
            (f"PINN {name}", pred[name], field_style),
            (f"|PINN - simulation| {name}", error, {"cmap": "viridis"}),
        )
        for ax, (title, values, style) in zip(row, panels, strict=True):
            mesh = ax.pcolormesh(test["x"], test["t"], values, **style)
            ax.set_title(title)
            fig.colorbar(mesh, ax=ax)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("t")
    fig.tight_layout()
    fig.savefig("plots/fields.png", dpi=200)


def plot_losses(model_path: str) -> None:
    """Training and validation losses of each loss term."""
    logs = torch.load(model_path, map_location="cpu", weights_only=True)["logs"]
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, log, title in (
        (ax_train, logs["epoch_train_losses_log"], "Training loss"),
        (ax_val, logs["epoch_val_losses_log"], "Validation loss"),
    ):
        for term in log[0]:
            ax.semilogy([entry[term] for entry in log], label=term)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
    ax_train.legend()
    fig.tight_layout()
    fig.savefig("plots/losses.png", dpi=200)


if __name__ == "__main__":
    example_name = "alfven_waves_1D"
    model_path = f"models/{example_name}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    pinn = pinninit.PINN.from_yaml(f"param_{example_name}.yml", device)
    pinn.load(model_path)

    # Evaluate it on the test set
    test = load_test_data("data/test.hdf5")
    u, b = evaluate_pinn(pinn, test["x"].ravel(), test["t"].ravel())
    pred = {"u": u.reshape(test["x"].shape), "b": b.reshape(test["x"].shape)}

    for name in ("u", "b"):
        error = relative_error(pred[name], test[name])
        print(f"Relative error of {name}: {error:.3e}")

    Path("plots").mkdir(exist_ok=True)
    plot_fields(test, pred)
    plot_losses(model_path)
    print("Plots saved to plots/")
