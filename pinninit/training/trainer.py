from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..loss.loss_term_group import LossTermGroup
    from ..nn.neural_network import NeuralNetwork


class Trainer:
    def __init__(
        self,
        model: NeuralNetwork,
        optimizer: torch.optim.Optimizer,
        loss_groups: list[LossTermGroup],
        device: torch.device,
    ) -> None:
        ### Note that e.g. train_loss_groups = [data_loss_group, col_loss_group]
        self.model = model
        self.optimizer = optimizer
        self.loss_groups = loss_groups
        self.device = device

    def step(
        self,
    ) -> None:
        # Zero the gradients
        self.optimizer.zero_grad()

        # Compute the loss of loss groups
        self.group_train_losses, self.group_val_losses = self.compute_group_losses(
            self.loss_groups
        )

        # Get sum of each of training and validation losses
        self.train_loss = torch.stack(self.group_train_losses).sum()
        self.val_loss = torch.stack(self.group_val_losses).sum()

        # Backpropagation
        self.train_loss.backward(retain_graph=True)  # type: ignore[no-untyped-call]

        # Update the model parameters
        self.optimizer.step()

    def train(self, N_iter: int) -> None:
        for iter in range(N_iter):
            self.iter = iter
            self.step()
            self.print_status()

    def compute_group_losses(
        self, loss_groups: list[LossTermGroup]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # Loop through all loss term groups
        group_train_losses = []
        group_val_losses = []
        for group in loss_groups:
            # Compute loss
            train_loss = group.compute_loss_train(self.model)
            val_loss = group.compute_loss_val(self.model)

            # Store loss for each loss term group
            group_train_losses.append(train_loss)
            group_val_losses.append(val_loss)

        return group_train_losses, group_val_losses

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
            ### temp

            train_losses = [loss.item() for loss in self.group_train_losses]
            val_losses = [loss.item() for loss in self.group_val_losses]

            print(
                f"Iter: {self.iter}, "
                f"Training Loss: {train_losses}, "
                f"Val Loss: {val_losses}, "
            )
