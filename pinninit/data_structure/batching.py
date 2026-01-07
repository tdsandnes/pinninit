from __future__ import annotations

from torch.utils.data import Dataset


class BatchingDataset(Dataset[int]):
    """Dataset that returns indices for batching"""

    def __init__(self, n_points: int):
        self.n_points = n_points

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, idx: int) -> int:
        return idx
