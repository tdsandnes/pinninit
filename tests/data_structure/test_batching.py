"""Unit tests for data_structure/batching.py."""

from __future__ import annotations

from collections.abc import Iterable

import pytest
import torch

from pinninit.data_structure.batching import Batches
from pinninit.data_structure.coords import Coords

from ..helpers import N_POINTS


def batch_indices(batches: Iterable[Coords]) -> list[list[int]]:
    """The indices of the points in each batch, within the full coords."""
    indices = []
    for batch in batches:
        assert batch.parent_idx is not None
        indices.append(batch.parent_idx.tolist())
    return indices


def one_pass(batches: Batches) -> list[list[int]]:
    """The indices of the points in each batch of the next pass."""
    batches.reset()
    return batch_indices(batches.next() for _ in range(len(batches)))


class TestBatches:
    def test_batches_cover_every_point_once(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=3)

        indices = one_pass(batches)

        assert [len(idx) for idx in indices] == [4, 3, 3]
        assert sorted(i for idx in indices for i in idx) == list(range(N_POINTS))

    def test_unshuffled_batches_are_in_index_order(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=3, shuffle=False)

        assert one_pass(batches) == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_each_pass_is_reshuffled(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=3)

        assert one_pass(batches) != one_pass(batches)

    def test_next_starts_a_new_pass_when_the_batches_run_out(
        self, coords_2d: Coords
    ) -> None:
        batches = Batches(coords_2d, n_batches=2, shuffle=False)

        indices = batch_indices(batches.next() for _ in range(3))

        assert indices == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]

    def test_reset_starts_a_new_pass(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=2, shuffle=False)
        batches.next()

        batches.reset()

        assert batch_indices([batches.next()]) == [[0, 1, 2, 3, 4]]

    def test_batches_are_linked_to_the_full_coords(self, coords_2d: Coords) -> None:
        batch = Batches(coords_2d, n_batches=2).next()

        assert batch.parent is coords_2d
        assert batch.get_coord("x").requires_grad

    def test_in_order(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=3)

        indices = batch_indices(batches.in_order())

        assert indices == [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def test_in_order_does_not_change_the_training_batches(
        self, coords_2d: Coords
    ) -> None:
        torch.manual_seed(0)
        batches = Batches(coords_2d, n_batches=3)
        torch.manual_seed(0)
        batches_with_evaluation = Batches(coords_2d, n_batches=3)

        list(batches_with_evaluation.in_order())

        assert one_pass(batches_with_evaluation) == one_pass(batches)

    def test_batching_leaves_the_global_rng_alone(self, coords_2d: Coords) -> None:
        batches = Batches(coords_2d, n_batches=3)
        state = torch.get_rng_state()

        one_pass(batches)
        list(batches.in_order())

        assert torch.equal(torch.get_rng_state(), state)

    def test_shuffling_follows_the_global_seed(self, coords_2d: Coords) -> None:
        torch.manual_seed(1)
        batches = Batches(coords_2d, n_batches=3)
        torch.manual_seed(1)
        same_seed_batches = Batches(coords_2d, n_batches=3)
        torch.manual_seed(2)
        other_seed_batches = Batches(coords_2d, n_batches=3)

        first_pass = one_pass(batches)
        assert one_pass(same_seed_batches) == first_pass
        assert one_pass(other_seed_batches) != first_pass

    def test_state_dict_restores_the_shuffling(self, coords_2d: Coords) -> None:
        # Saved after one pass, so the state differs from a new Batches' state
        batches = Batches(coords_2d, n_batches=3)
        one_pass(batches)
        state = batches.state_dict()
        expected = one_pass(batches)

        restored = Batches(coords_2d, n_batches=3)
        restored.load_state_dict(state)

        assert one_pass(restored) == expected

    def test_n_batches_is_capped_at_n_points(self, coords_2d: Coords) -> None:
        assert len(Batches(coords_2d, n_batches=N_POINTS + 5)) == N_POINTS

    def test_n_batches_must_be_positive(self, coords_2d: Coords) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Batches(coords_2d, n_batches=0)
