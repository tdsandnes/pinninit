from __future__ import annotations

import sys
from pathlib import Path

import torch

# Import pinninit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import pinninit


def train() -> None:
    # Configuration
    example_name = "fournier07_1D"

    param_path = f"param_{example_name}.yml"
    model_path = f"models/{example_name}.pth"

    # Ensure output directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init PINN
    pinn = pinninit.PINN(param_path, device)

    # Train
    pinn.train()

    # Save model
    pinn.save(model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train()
