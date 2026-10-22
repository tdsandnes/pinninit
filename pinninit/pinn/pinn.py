from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..config.pinn_config import PINNConfig

if TYPE_CHECKING:
    import torch.nn as nn

    from ..data_structure.coords import Coords
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
        trainer: Trainer,
    ) -> None:
        self.config = config
        self.device = device
        self.model = model
        self.loss_containers = loss_containers
        self.learnt_consts = learnt_consts
        self.trainer = trainer

    @property
    def optimizer(self) -> Optimizer:
        """The optimizer training uses now (a trainer module can replace it,
        e.g. UpdateOptimizer)."""
        return self.trainer.optimizer

    @classmethod
    def from_yaml(
        cls,
        param_file_path: str | Path,
        device: torch.device,
    ) -> PINN:
        """Build a PINN from a parameter file. Sets torch's default dtype to
        the file's dtype (float64 unless set) and, if the file sets
        random_seed, seeds torch's random number generator."""
        config = PINNConfig.from_yaml(param_file_path)

        # Everything is built and trained in this precision
        torch.set_default_dtype(config.dtype)

        # Set torch seed from the parameter file (random_seed: random -> unseeded)
        if config.random_seed is not None:
            torch.manual_seed(config.random_seed)

        # Relative paths in the config resolve against the parameter file
        base_dir = Path(param_file_path).parent

        from ..builders.pinn_builder import build_pinn

        return build_pinn(config, device, base_dir=base_dir)

    def train(self) -> None:
        self.trainer.train()

    def evaluate_model(self, coords: Coords) -> torch.Tensor:
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
        """Save the model, the training logs and the state to resume training
        from. After training, the model is the best model."""
        self.trainer.save(path)

    def load(self, path: str | Path) -> None:
        """Load what save saved. Calling train then resumes training."""
        self.trainer.load(path)
