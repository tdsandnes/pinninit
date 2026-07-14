from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..config.pinn_config import PINNConfig

if TYPE_CHECKING:
    import torch.nn as nn

    from ..data_structure.coords_dataclass import CoordsProtocol
    from ..loss.loss_term_container import LossTermContainer
    from ..nn.neural_network import NeuralNetwork
    from ..optimizer.optimizer import Optimizer
    from ..trainer.trainer import Trainer


class PINN:
    """Physics-informed neural network."""

    def __init__(
        self,
        config: PINNConfig,
        device: torch.device,
        model: NeuralNetwork,
        loss_containers: dict[str, LossTermContainer],
        learnt_consts: dict[str, nn.Parameter],
        optimizer: Optimizer,
        trainer: Trainer,
    ) -> None:
        self.config = config
        self.device = device
        self.model = model
        self.loss_containers = loss_containers
        self.learnt_consts = learnt_consts
        self.optimizer = optimizer
        self.trainer = trainer

    @classmethod
    def from_yaml(
        cls,
        param_file_path: str | Path,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
        seed: int | None = 12345,
    ) -> PINN:
        # Set default dtype to higher precision
        torch.set_default_dtype(dtype)

        # Set torch seed
        if seed is not None:
            torch.manual_seed(seed)

        config = PINNConfig.from_yaml(param_file_path)

        # Relative paths in the config resolve against the parameter file
        base_dir = Path(param_file_path).parent

        from ..builders.pinn_builder import build_pinn

        return build_pinn(config, device, base_dir=base_dir)

    def train(self) -> None:
        self.trainer.train()

    def evaluate_model(self, coords: CoordsProtocol) -> torch.Tensor:
        """Evaluate the model at the given coords."""
        self.model.eval()
        with torch.no_grad():
            output = self.model.evaluate(coords).detach()
        self.model.train()
        return output

    def evaluate_unweighted_val_losses(self) -> dict[str, float]:
        """Evaluate the unweighted validation loss of each loss term."""
        self.model.eval()
        val_losses = {
            label: container.compute_full_val_loss(self.model).item()
            for label, container in self.loss_containers.items()
        }
        self.model.train()
        return val_losses

    def evaluate_train_residuals(self) -> dict[str, torch.Tensor]:
        """Evaluate per-point residuals of each loss term on its training set."""
        self.model.eval()
        residuals = {
            label: container.compute_train_batch_loss_and_residuals(
                self.model, container.train_eval_context.coords, detach=True
            )[1]
            for label, container in self.loss_containers.items()
        }
        self.model.train()
        return residuals

    def save(self, path: str | Path) -> None:
        trainer = self.trainer
        logger = trainer.logger
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch_train_losses_log": logger.epoch_train_losses_log,
                "epoch_val_losses_log": logger.epoch_val_losses_log,
                "epoch_loss_term_weights_log": logger.epoch_loss_term_weights_log,
                "epoch_learnt_consts_log": logger.epoch_learnt_consts_log,
                "events": logger.events,
                "best_model_epoch": trainer.best_model_epoch,
                "best_geometric_mean_val_loss": trainer.best_geometric_mean_val_loss,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"No such file: '{path}'")

        checkpoint = torch.load(path, map_location=self.device)
        missing, _unexpected = self.model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing:
            raise RuntimeError(f"Checkpoint is missing model parameters: {missing}")

        # Restore logs onto the trainer logger
        logger = self.trainer.logger
        logger.epoch_train_losses_log = checkpoint.get("epoch_train_losses_log", [])
        logger.epoch_val_losses_log = checkpoint.get("epoch_val_losses_log", [])
        logger.epoch_loss_term_weights_log = checkpoint.get(
            "epoch_loss_term_weights_log", []
        )
        logger.epoch_learnt_consts_log = checkpoint.get("epoch_learnt_consts_log", [])
        logger.events = checkpoint["events"]
        self.trainer.best_model_epoch = checkpoint["best_model_epoch"]
        self.trainer.best_geometric_mean_val_loss = checkpoint.get(
            "best_geometric_mean_val_loss"
        )
