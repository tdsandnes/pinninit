from collections.abc import Iterator
from typing import cast

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .coords_dataclass import CoordsProtocol


class BatchDataset(Dataset[int]):
    """Dataset that returns indices for batching"""

    def __init__(self, n_points: int):
        self.n_points = n_points

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> int:
        return idx


class EqualSplitBatchSampler(Sampler[list[int]]):
    """Yields exactly n_batches index batches per pass, with sizes differing
    by at most one point, reshuffled on each new pass."""

    def __init__(self, n_points: int, n_batches: int, shuffle: bool = True) -> None:
        if n_batches <= 0:
            raise ValueError(f"n_batches must be positive, got {n_batches}")
        if n_batches > n_points:
            raise ValueError(
                f"n_batches ({n_batches}) cannot exceed n_points ({n_points})"
            )
        self.n_points = n_points
        self.n_batches = n_batches
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            idx = torch.randperm(self.n_points)
        else:
            idx = torch.arange(self.n_points)
        for chunk in torch.tensor_split(idx, self.n_batches):
            yield chunk.tolist()


def make_dataloader(
    coords: CoordsProtocol,
    n_batches: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader[CoordsProtocol]:
    """Create a DataLoader that yields exactly n_batches batches per pass,
    together covering every point exactly once."""
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    # A loss term with fewer points than n_batches cannot fill that many
    # non-empty batches; it cycles more than once per epoch instead
    n_batches = min(n_batches, coords.n_points)

    dataset = BatchDataset(coords.n_points)

    # Define collate_fn to convert list of indices into a batched coordinates
    def collate_fn(idx_list: list[int]) -> CoordsProtocol:
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=coords.device)
        return coords.get_batch(idx_tensor)

    loader = DataLoader(
        dataset=dataset,
        batch_sampler=EqualSplitBatchSampler(coords.n_points, n_batches, shuffle),
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
