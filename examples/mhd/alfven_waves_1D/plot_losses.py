from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_PATH = "models/alfven_waves_1D.pth"
STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Legend label and colour of each loss term
TERMS = {
    "data_b": ("data loss", "#00798c"),
    "zp_eqn": (r"$z^+$ loss", "#d1495b"),
    "zm_eqn": (r"$z^-$ loss", "#edae49"),
    "dbdx_bc": ("bc loss", "#66a182"),
}


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
        for term, (label, color) in TERMS.items():
            losses = [entry[term] for entry in log]
            ax.plot(epochs, losses, color=color, label=label, alpha=0.8)

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

    ax_train.set_yscale("log")
    ax_train.set_ylabel("Training loss")
    ax_val.set_ylabel("Validation loss")
    ax_val.set_xlabel("Epoch")
    ax_val.legend(loc="upper right")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/losses.png")
