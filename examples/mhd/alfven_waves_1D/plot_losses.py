from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_PATH = "models/alfven_waves_1D.pth"
STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Legend label and colour of each loss term
TERMS = {
    "data_b": ("data loss", "#5e5e6d"),
    "zp_eqn": (r"$z^+$ loss", "#2e7d5b"),
    "zm_eqn": (r"$z^-$ loss", "#4fb08a"),
    "dbdx_bc": ("bc loss", "#8c5a3f"),
}

# The raw per-epoch losses are drawn faintly, with a rolling geometric mean on top
RAW_LOSS_ALPHA = 0.25
RAW_LOSS_LINEWIDTH = 0.4
SMOOTH_LOSS_LINEWIDTH = 0.9
# Width of the rolling window, in epochs
SMOOTH_WINDOW_EPOCHS = 1000


def rolling_geometric_mean(values: list[float], window: int) -> np.ndarray:
    """Geometric mean over a centred window, which shrinks at the ends."""
    log_values = np.log(np.asarray(values))
    # A window longer than the run would make np.convolve return extra values
    kernel = np.ones(min(window, len(log_values)))
    total = np.convolve(log_values, kernel, mode="same")
    count = np.convolve(np.ones_like(log_values), kernel, mode="same")
    return np.exp(total / count)


if __name__ == "__main__":
    plt.style.use(STYLE_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    logs = checkpoint["logs"]
    train_log = logs["epoch_train_losses_log"]
    val_log = logs["epoch_val_losses_log"]

    fig, (ax_train, ax_val) = plt.subplots(
        2, 1, figsize=(6.5, 4.5), sharex=True, sharey=True, layout="constrained"
    )

    # An epoch's training loss is the mean over its steps, each from before
    # the step's update, so it is plotted at the epoch's start. Its validation
    # loss is from the epoch's end
    for ax, log, first_epoch in ((ax_train, train_log, 0), (ax_val, val_log, 1)):
        epochs = np.arange(len(log)) + first_epoch
        for i, (term, (label, color)) in enumerate(TERMS.items()):
            losses = [entry[term] for entry in log]
            zorder = len(TERMS) - i
            # Rasterised: tens of thousands of points per line bloat a PDF otherwise
            ax.plot(
                epochs,
                losses,
                color=color,
                alpha=RAW_LOSS_ALPHA,
                linewidth=RAW_LOSS_LINEWIDTH,
                zorder=zorder,
                rasterized=True,
            )
            ax.plot(
                epochs,
                rolling_geometric_mean(losses, SMOOTH_WINDOW_EPOCHS),
                color=color,
                label=label,
                linewidth=SMOOTH_LOSS_LINEWIDTH,
                zorder=zorder + len(TERMS),
            )

    # The best model is the one with the lowest geometric mean validation loss
    # after warmup, as pinninit doesn't track the best model while warming up
    warmup_epochs = checkpoint["parameters"]["Trainer"].get("warmup_epochs", 0)
    val_losses = np.array([[entry[term] for term in TERMS] for entry in val_log])
    geometric_mean = np.exp(np.log(val_losses).mean(axis=1))
    val_epochs = np.arange(1, len(val_log) + 1)
    after_warmup = val_epochs > warmup_epochs
    ax_val.plot(
        val_epochs[after_warmup],
        np.minimum.accumulate(geometric_mean[after_warmup]),
        color="black",
        ls="--",
        label="lowest geometric mean",
    )

    for letter, ax in zip("ab", (ax_train, ax_val), strict=True):
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, va="bottom")

    ax_train.set_yscale("log")
    ax_train.set_ylabel("Training loss")
    ax_val.set_ylabel("Validation loss")
    ax_val.set_xlabel("Training epoch")
    ax_val.legend(loc="upper right")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/losses.png")
