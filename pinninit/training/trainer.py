from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..loss.loss_term_group import LossTermCoordGroup
    from ..nn.neural_network import NeuralNetwork


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        optimizer: torch.optim.Optimizer,
        train_loss_groups: list[LossTermCoordGroup],
        val_loss_groups: list[LossTermCoordGroup],
        test_loss_groups: list[LossTermCoordGroup],
        device: torch.device,
    ) -> None:
        ### Note that e.g. train_loss_groups = [data_loss_group, col_loss_group]
        self.model = model
        self.optimizer = optimizer
        self.train_loss_groups = train_loss_groups
        self.val_loss_groups = val_loss_groups
        self.test_loss_groups = test_loss_groups
        self.device = device

    def step(
        self,
    ) -> None:
        # Zero the gradients
        self.optimizer.zero_grad()

        # Compute the loss of training, validation, and test Coord groups
        self.train_loss, self.train_group_losses = self.compute_loss(
            self.train_loss_groups
        )
        self.val_loss, self.val_group_losses = self.compute_loss(self.val_loss_groups)
        self.test_loss, self.test_group_losses = self.compute_loss(
            self.test_loss_groups
        )

        # Backpropagation
        self.train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        # Update the model parameters
        self.optimizer.step()

    def train(self, N_iter: int) -> None:
        for iter in range(N_iter):
            self.iter = iter
            self.step()
            self.print_status()

    def compute_loss(
        self, loss_groups: list[LossTermCoordGroup]
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # Loop through all loss term groups
        group_losses = []
        for group in loss_groups:
            group.apply_model(
                self.model,
            )

            loss = group.compute_loss()

            # Store loss for each training loss term group
            group_losses.append(loss)

        # Sum for total loss of given groups
        total_loss = torch.stack(group_losses).sum()

        return total_loss, group_losses

    ###    def update_optimizer(self,):
    ### Directs to a polymorphic UpdateOptimizer class in a update_optimizer.py

    ###    def update_learning_rate(self,):
    ### Directs to a polymorphic UpdateLearningRate class in a update_learning_rate.py

    ###    def update_loss_term_weights(self,):
    ##  polymorphic UpdateLossTermWeights class in a update_loss_term_weights.py

    ##    def early_stopping(self,):
    ### Directs to a polymorphic EarlyStopping class in a early_stopping.py

    def print_status(self, verbosity: int = 1) -> None:
        if self.iter % 500 == 0:
            train_losses = [loss.item() for loss in self.train_group_losses]
            val_losses = [loss.item() for loss in self.val_group_losses]
            test_losses = [loss.item() for loss in self.test_group_losses]

            print(
                f"Iter: {self.iter}, "
                f"Training Loss: {train_losses}, "
                f"Val Loss: {val_losses}, "
                f"Test Loss: {test_losses}"
            )
