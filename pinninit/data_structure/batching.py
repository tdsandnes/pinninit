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
