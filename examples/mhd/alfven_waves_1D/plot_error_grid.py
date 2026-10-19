from __future__ import annotations

import sys
from pathlib import Path

import colorcet as cc
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from alfven_waves_1D_equations import u_eqn
from matplotlib.colors import LogNorm

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords, Operators, make_coords_dataclass

PARAM_PATH = "param_alfven_waves_1D.yml"
MODEL_PATH = "models/alfven_waves_1D.pth"
STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Lines t = 21 - x and t = 29 + x, as (t at x = 0, slope dt/dx)
ALFVEN_SPEED_LINES = ((21.0, -1.0), (29.0, 1.0))


def evaluate_pinn(
    pinn: pinninit.PINN, x: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The PINN's u and b at the points (x, t)."""
    coord_system = CartesianCoords(coord_axes={"x"}, device=pinn.device)
    XTCoords = make_coords_dataclass({"x", "t"}, coord_system, pinn.device)
    coords = XTCoords(
        x=torch.tensor(x, device=pinn.device), t=torch.tensor(t, device=pinn.device)
    )
    out = pinn.evaluate_model(coords)
    labels = pinn.model.output_field_labels
    nn_out = out[:, labels.index("nn_out")]
    u = u_eqn(nn_out, coords.get_coord("x"), Operators(coords))
    b = out[:, labels.index("b")]
    return u.cpu().numpy(), b.cpu().numpy()


if __name__ == "__main__":
    plt.style.use(STYLE_PATH)

    # Load the test set, as (t, x) grids
    with h5py.File("data/test.hdf5", "r") as f:
        group = f["test"]["test"]
        test = {name: group["Coords"][name][:] for name in ("x", "t")}
        test |= {name: group["Fields"][name][:] for name in ("u", "b")}
    n_x = len(np.unique(test["x"]))
    test = {name: values.reshape(-1, n_x) for name, values in test.items()}

    # Evaluate the trained model on it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn = pinninit.PINN.from_yaml(PARAM_PATH, device)
    pinn.load(MODEL_PATH)
    u, b = evaluate_pinn(pinn, test["x"].ravel(), test["t"].ravel())
    pred = {"u": u.reshape(test["x"].shape), "b": b.reshape(test["x"].shape)}

    errors = {}
    for field in ("u", "b"):
        errors[field] = np.abs(pred[field] - test[field])
        relative = np.linalg.norm(errors[field]) / np.linalg.norm(test[field])
        print(f"Relative L2 error of {field}: {relative:.3e}")

    # u = 0 at x = 1 by construction, so its error there is only round-off. It
    # is left out of the colour range and drawn black
    errors["u"] = np.ma.masked_array(errors["u"])
    errors["u"][:, -1] = np.ma.masked
    values = np.ma.concatenate([errors["u"].ravel(), errors["b"].ravel()])
    norm = LogNorm(vmin=values.min(), vmax=values.max())
    cmap = cc.cm.kbgyw.copy()
    cmap.set_bad("black")

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 3.0), sharey=True, layout="constrained"
    )
    extent = (0.0, 1.0, test["t"].min(), test["t"].max())
    for ax, field, letter in zip(axes, ("u", "b"), "ab", strict=True):
        image = ax.imshow(
            errors[field],
            origin="lower",
            extent=extent,
            norm=norm,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        for t_at_x0, slope in ALFVEN_SPEED_LINES:
            ax.plot(
                [0, 1], [t_at_x0, t_at_x0 + slope], color="black", ls="--", alpha=0.6
            )
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel=r"$x$")
        ax.set_title(f"({letter})", loc="left")
        ax.set_title(rf"${field}$", loc="right")
        ax.minorticks_off()
    axes[0].set_ylabel(r"$t$")
    fig.colorbar(image, ax=axes, label="Absolute error")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/test_error_grid.png")
