from __future__ import annotations

import string
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from fournier07_1D_equations import b_eqn, u_eqn
from sim_config import Config

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit
from pinninit import CartesianCoords, Operators, make_coords_dataclass

PARAM_PATH = "param_fournier07_1D.yml"
MODEL_PATH = "models/fournier07_1D.pth"
STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Times to plot, one column each. They have to be snapshot times
TIMES = [0.05, 0.1, 0.15, 0.2]
PINN_COLOR = "#ab008b"


def evaluate_pinn(
    pinn: pinninit.PINN, x: np.ndarray, t: np.ndarray
) -> dict[str, np.ndarray]:
    """The PINN's u and b at the points (x, t)."""
    coord_system = CartesianCoords(coord_axes={"x"}, device=pinn.device)
    XTCoords = make_coords_dataclass({"x", "t"}, coord_system, pinn.device)
    coords = XTCoords(
        x=torch.tensor(x, device=pinn.device), t=torch.tensor(t, device=pinn.device)
    )
    out = pinn.evaluate_model(coords)
    labels = pinn.model.output_field_labels
    x_coord, operators = coords.get_coord("x"), Operators(coords)
    u = u_eqn(out[:, labels.index("nn_u")], x_coord, operators)
    b = b_eqn(out[:, labels.index("nn_b")], x_coord, operators)
    return {"u": u.cpu().numpy(), "b": b.cpu().numpy()}


if __name__ == "__main__":
    plt.style.use(STYLE_PATH)

    # Load the simulation snapshots and the trained model
    with h5py.File(Config().snapshot_path, "r") as f:
        snapshot_times, x = f["t"][:], f["x"][:]
        sim = {"u": f["u"][:], "b": f["b"][:]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn = pinninit.PINN.from_yaml(PARAM_PATH, device)
    pinn.load(MODEL_PATH)

    fig, axes = plt.subplots(
        2,
        len(TIMES),
        figsize=(6.5, 2.6),
        sharex=True,
        sharey="row",
        layout="constrained",
    )
    for col, time in enumerate(TIMES):
        idx = int(np.argmin(np.abs(snapshot_times - time)))
        if not np.isclose(snapshot_times[idx], time):
            raise ValueError(f"No snapshot at t = {time}")
        pred = evaluate_pinn(pinn, x, np.full_like(x, time))

        for row, field in enumerate(("u", "b")):
            ax = axes[row, col]
            ax.plot(x, sim[field][idx], color="black", lw=0.8, label="simulation")
            ax.plot(x, pred[field], color=PINN_COLOR, ls="--", lw=0.8, label="PINN")
            letter = string.ascii_lowercase[row * len(TIMES) + col]
            ax.text(0.05, 0.95, f"({letter})", transform=ax.transAxes, va="top")

        axes[0, col].set_title(rf"$t = {time:g}$")
        axes[1, col].set_xlabel(r"$x$")

    axes[0, 0].set(xlim=(-1, 1), ylabel=r"$u$")
    axes[1, 0].set_ylabel(r"$b$")
    axes[1, -1].legend(loc="lower right")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/snapshots.png")
