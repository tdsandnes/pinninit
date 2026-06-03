"""Unit tests for data pipeline batching utilities."""

from __future__ import annotations

import pytest
import torch

from pinninit.data_structure.batching import (
    BatchDataset,
    make_dataloader,
)
from pinninit.data_structure.coords_dataclass import (
    CoordsProtocol,
    make_coords_dataclass,
)
from pinninit.geometry.coordinate_system import CoordSystem

from .fixtures import (
    make_cartesian_2d,
    make_device,
)


# Local fixtures matching existing suite style
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def cartesian_2d(device: torch.device) -> CoordSystem:
    return make_cartesian_2d(device)


# Local Helper to generate coords instances
def build_coords(
    coord_system: CoordSystem,
    n_points: int,
    device: torch.device,
) -> CoordsProtocol:
    Coords = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=device,
    )
    x_val = torch.linspace(0.0, 10.0, n_points, requires_grad=True, device=device)
    y_val = torch.linspace(10.0, 20.0, n_points, requires_grad=True, device=device)
    return Coords(coord_system=coord_system, x=x_val, y=y_val)


class TestBatchDatasetProperties:
    def test_dataset_index_passthrough(self) -> None:
        n_points = 50
        dataset = BatchDataset(n_points=n_points)

        assert len(dataset) == n_points
        assert dataset[0] == 0
        assert dataset[n_points - 1] == n_points - 1


class TestMakeDataLoaderBehavior:
    def test_dataloader_batch_grouping_and_types(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
    ) -> None:
        n_points = 50
        batch_size = 15
        coords = build_coords(cartesian_2d, n_points=n_points, device=device)

        # Original reference tensors
        original_x = coords.get_coord("x")
        original_y = coords.get_coord("y")

        loader = make_dataloader(
            coords=coords,
            batch_size=batch_size,
            shuffle=False,
        )

        batches = list(loader)

        # Verify that the custom collate fn outputs the real coords protocol dataclass
        assert all(isinstance(batch, type(coords)) for batch in batches)

        # Expected lengths for 50 elements with batch_size 15 -> [15, 15, 15, 5]
        assert len(batches) == 4
        assert [batch.n_points for batch in batches] == [15, 15, 15, 5]

        # Compare batch coordinate values with their expected sequential slices
        assert torch.allclose(batches[0].get_coord("x"), original_x[0:15])
        assert torch.allclose(batches[0].get_coord("y"), original_y[0:15])
        assert torch.allclose(batches[1].get_coord("x"), original_x[15:30])
        assert torch.allclose(batches[1].get_coord("y"), original_y[15:30])
        assert torch.allclose(batches[2].get_coord("x"), original_x[30:45])
        assert torch.allclose(batches[2].get_coord("y"), original_y[30:45])
        assert torch.allclose(batches[3].get_coord("x"), original_x[45:50])
        assert torch.allclose(batches[3].get_coord("y"), original_y[45:50])

    def test_dataloader_shuffling_distribution(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        n_points = 50
        batch_size = 15
        coords = build_coords(cartesian_2d, n_points=n_points, device=device)

        # Original reference tensors
        original_x = coords.get_coord("x")
        original_y = coords.get_coord("y")

        torch.manual_seed(0)
        loader = make_dataloader(
            coords=coords,
            batch_size=batch_size,
            shuffle=True,
        )

        # Collect component tensors across all batches
        shuffled_x_list = []
        shuffled_y_list = []
        for batch in loader:
            shuffled_x_list.append(batch.get_coord("x"))
            shuffled_y_list.append(batch.get_coord("y"))
        all_shuffled_x = torch.cat(shuffled_x_list, dim=0)
        all_shuffled_y = torch.cat(shuffled_y_list, dim=0)

        assert len(all_shuffled_x) == n_points
        assert len(all_shuffled_y) == n_points
        assert torch.allclose(
            torch.sort(all_shuffled_x).values, torch.sort(original_x).values
        )
        assert torch.allclose(
            torch.sort(all_shuffled_y).values, torch.sort(original_y).values
        )
        assert not torch.equal(all_shuffled_x, original_x)

    def test_empty_dataset_produces_no_batches(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
    ) -> None:
        coords = build_coords(cartesian_2d, n_points=0, device=device)

        loader = make_dataloader(
            coords=coords,
            batch_size=10,
            shuffle=False,
        )

        assert list(loader) == []

    def test_batch_preserves_device(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
    ) -> None:
        coords = build_coords(cartesian_2d, n_points=20, device=device)

        loader = make_dataloader(
            coords=coords,
            batch_size=10,
            shuffle=False,
        )

        batch = next(iter(loader))

        assert batch.device == coords.device
        assert batch.get_coord("x").device == coords.device
        assert batch.get_coord("y").device == coords.device

    def test_invalid_batch_size_raises(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
    ) -> None:
        coords = build_coords(cartesian_2d, n_points=10, device=device)

        with pytest.raises(ValueError, match="batch_size"):
            make_dataloader(coords=coords, batch_size=0)

    def test_negative_num_workers_raises(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
    ) -> None:
        coords = build_coords(cartesian_2d, n_points=10, device=device)

        with pytest.raises(ValueError, match="num_workers"):
            make_dataloader(
                coords=coords,
                batch_size=5,
                num_workers=-1,
            )


class TestBatchAutogradGraphPreservation:
    def test_collate_yields_differentiable_leaves(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        n_points = 20
        coords = build_coords(cartesian_2d, n_points=n_points, device=device)

        loader = make_dataloader(
            coords=coords,
            batch_size=10,
            shuffle=False,
        )

        batch_coords = next(iter(loader))

        assert batch_coords.is_batch is True
        assert batch_coords.get_coord("x").requires_grad is True
        assert batch_coords.get_coord("x").is_leaf is True

        loss = (batch_coords.get_coord("x") ** 2).sum()
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=batch_coords.get_coord("x"),
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert list(grads.shape) == list(batch_coords.get_coord("x").shape)
        assert torch.allclose(grads, 2 * batch_coords.get_coord("x"))
