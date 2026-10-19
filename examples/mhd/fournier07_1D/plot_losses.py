from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

MODEL_PATH = "models/fournier07_1D.pth"
STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Legend label and colour of each loss term
TERMS = {
    "data_b": ("data loss", "#00798c"),
    "momentum_eqn": ("momentum loss", "#d1495b"),
    "induction_eqn": ("induction loss", "#edae49"),
}


if __name__ == "__main__":
    plt.style.use(STYLE_PATH)
    logs = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)["logs"]
    train_log = logs["epoch_train_losses_log"]
    val_log = logs["epoch_val_losses_log"]

    fig, (ax_train, ax_val) = plt.subplots(
        2, 1, figsize=(6.5, 4.5), sharex=True, sharey=True, layout="constrained"
    )

    # pinninit logs training losses from epoch 0 and validation from epoch 1
    for ax, log, first_epoch in ((ax_train, train_log, 0), (ax_val, val_log, 1)):
        epochs = np.arange(len(log)) + first_epoch
        for term, (label, color) in TERMS.items():
            losses = [entry[term] for entry in log]
            ax.plot(epochs, losses, color=color, label=label, alpha=0.8)

    # The best model is the one with the lowest geometric mean validation loss
    val_losses = np.array([[entry[term] for term in TERMS] for entry in val_log])
    geometric_mean = np.exp(np.log(val_losses).mean(axis=1))
    ax_val.plot(
        np.arange(1, len(val_log) + 1),
        np.minimum.accumulate(geometric_mean),
        color="black",
        ls="--",
        label="lowest geometric mean",
    )

    ax_train.set_yscale("log")
    ax_train.set_ylabel("Training loss")
    ax_val.set_ylabel("Validation loss")
    ax_val.set_xlabel("Training Epoch")
    ax_val.legend(loc="upper right")

    Path("plots").mkdir(exist_ok=True)
    fig.savefig("plots/losses.png")
