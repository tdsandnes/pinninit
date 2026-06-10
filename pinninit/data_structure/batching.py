from collections.abc import Iterator
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset

from .coords_dataclass import CoordsProtocol


class BatchDataset(Dataset[int]):
    """Dataset that returns indices for batching"""

    def __init__(self, n_points: int):
        self.n_points = n_points

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> int:
        return idx


def make_dataloader(
    coords: CoordsProtocol,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[CoordsProtocol]:
    """Utility to create a DataLoader."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = BatchDataset(coords.n_points)

    # Define collate_fn to convert list of indices into a batched coordinates
    def collate_fn(idx_list: list[int]) -> CoordsProtocol:
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=coords.device)
        return coords.get_batch(idx_tensor)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Return the DataLoader with the correct type annotation
    return cast(DataLoader[CoordsProtocol], loader)


def iter_batches(
    dataloader: DataLoader[CoordsProtocol],
) -> Iterator[CoordsProtocol]:
    """Typed boundary around DataLoader iteration"""
    yield from dataloader


class BatchCycler:
    """Cycles indefinitely through a dataloader's batches, restarting (and
    reshuffling) each time the dataloader is exhausted."""

    def __init__(self, dataloader: DataLoader[CoordsProtocol]) -> None:
        self.dataloader = dataloader
        self._iter: Iterator[CoordsProtocol] | None = None

    def __len__(self) -> int:
        return len(self.dataloader)

    def next(self) -> CoordsProtocol:
        if self._iter is None:
            self._iter = iter_batches(self.dataloader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter_batches(self.dataloader)
            return next(self._iter)

    def reset(self) -> None:
        """Restart cycling from the beginning of the dataloader."""
        self._iter = None
