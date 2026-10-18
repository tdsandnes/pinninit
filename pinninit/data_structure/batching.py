from __future__ import annotations

from collections.abc import Iterator

import torch

from .coords import Coords


class Batches:
    """
    Splits coords into n_batches batches that together cover every point
    once, with sizes differing by at most one point. Each pass over the
    batches is reshuffled if shuffle is True.
    """

    def __init__(self, coords: Coords, n_batches: int, shuffle: bool = True) -> None:
        if n_batches <= 0:
            raise ValueError(f"n_batches must be positive, got {n_batches}")
        if coords.n_points == 0:
            raise ValueError("Coords must have at least one point to be batched")
        self.coords = coords
        # Coords with fewer points than n_batches cannot fill that many batches
        self.n_batches = min(n_batches, coords.n_points)
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(torch.initial_seed())
        # Indices of the batches left in the current pass
        self._pass: list[torch.Tensor] = []

    def __len__(self) -> int:
        return self.n_batches

    def next(self) -> Coords:
        """The next batch, starting a new pass when the current one runs out."""
        if not self._pass:
            self._pass = list(self._split(self.shuffle))
        return self.coords.get_batch(self._pass.pop(0))

    def reset(self) -> None:
        """Start a new pass at the next batch."""
        self._pass = []

    def in_order(self) -> Iterator[Coords]:
        """One pass over the batches in index order, e.g. for a full-set loss.

        Doesn't use the random number generator, so it doesn't change the
        training batches.
        """
        for idx in self._split(shuffle=False):
            yield self.coords.get_batch(idx)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """The shuffling state. Saved at the start of a pass, it is enough to
        give the same batches from then on."""
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        # A checkpoint loaded onto a GPU has its tensors there
        self.generator.set_state(state["generator"].cpu())

    def _split(self, shuffle: bool) -> tuple[torch.Tensor, ...]:
        n_points = self.coords.n_points
        if shuffle:
            idx = torch.randperm(n_points, generator=self.generator)
        else:
            idx = torch.arange(n_points)
        return torch.tensor_split(idx.to(self.coords.device), self.n_batches)
