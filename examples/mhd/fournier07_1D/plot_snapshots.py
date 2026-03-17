from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from sim_config import Config


def read_snapshot(path: Path) -> tuple[float, list[float], list[float], list[float]]:
    with h5py.File(path, "r") as f:
        t = f["Parameters"].attrs["t"]
        x = f["Coords"]["x"][:]
        u = f["Fields"]["u"][:]
        b = f["Fields"]["b"][:]
    return t, x, u, b


if __name__ == "__main__":
    snapshot_dir = Path(Config().snapshot_dir)
    files = sorted(snapshot_dir.glob("snapshot_*.hdf5"))

    t0, x0, u0, b0 = read_snapshot(files[0])
    t1, x1, u1, b1 = read_snapshot(files[-1])

    fig, (ax_u, ax_b) = plt.subplots(1, 2, figsize=(10, 4))

    ax_u.plot(x0, u0, label=f"t={t0:.3f}")
    ax_u.plot(x1, u1, label=f"t={t1:.3f}")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("u")
    ax_u.legend()

    ax_b.plot(x0, b0, label=f"t={t0:.3f}")
    ax_b.plot(x1, b1, label=f"t={t1:.3f}")
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("b")
    ax_b.legend()

    plt.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)  # creates directory if needed

    plt.savefig(plots_dir / "snapshot_plot.png", dpi=300)
