"""Unit tests for geometry/metric_tensor.py."""

from __future__ import annotations

import pytest
import torch

from pinninit.geometry.metric_tensor import (
    DiagonalMetricTensor,
    FullMetricTensor,
    IdentityMetricTensor,
)


class TestIdentityMetricTensor:
    def test_everything_is_unchanged(self, device: torch.device) -> None:
        metric = IdentityMetricTensor(4, device)
        v = torch.randn(4, 3, device=device)
        s = torch.randn(4, device=device)

        assert torch.equal(metric.sqrt_det, torch.ones(4, device=device))
        assert torch.equal(metric.inv_metric_times_vec(v), v)
        assert torch.equal(metric.sqrt_det_times_vec(v), v)
        assert torch.equal(metric.inv_sqrt_det_times_scalar(s), s)
        assert torch.equal(metric.scale_factors_times_vec(v), v)
        assert torch.equal(metric.inv_scale_factors_times_vec(v), v)


class TestDiagonalMetricTensor:
    def test_sqrt_det(self, device: torch.device) -> None:
        metric = DiagonalMetricTensor(
            torch.tensor([[1.0, 4.0], [9.0, 16.0]], device=device)
        )

        assert torch.allclose(metric.sqrt_det, torch.tensor([2.0, 12.0], device=device))

    def test_inv_metric_times_vec(self, device: torch.device) -> None:
        metric = DiagonalMetricTensor(
            torch.tensor([[1.0, 4.0], [9.0, 16.0]], device=device)
        )
        v = torch.ones(2, 2, device=device)

        expected = torch.tensor([[1.0, 1 / 4], [1 / 9, 1 / 16]], device=device)
        assert torch.allclose(metric.inv_metric_times_vec(v), expected)

    def test_scale_factors_are_the_square_roots_of_the_diagonal(
        self, device: torch.device
    ) -> None:
        metric = DiagonalMetricTensor(
            torch.tensor([[1.0, 4.0], [9.0, 16.0]], device=device)
        )
        v = torch.ones(2, 2, device=device)

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        assert torch.allclose(metric.scale_factors_times_vec(v), expected)
        assert torch.allclose(metric.inv_scale_factors_times_vec(v), 1 / expected)

    def test_sqrt_det_products(self, device: torch.device) -> None:
        metric = DiagonalMetricTensor(
            torch.tensor([[1.0, 4.0], [9.0, 16.0]], device=device)
        )
        v = torch.ones(2, 2, device=device)
        s = torch.ones(2, device=device)

        expected = torch.tensor([[2.0, 2.0], [12.0, 12.0]], device=device)
        assert torch.allclose(metric.sqrt_det_times_vec(v), expected)
        assert torch.allclose(
            metric.inv_sqrt_det_times_scalar(s),
            torch.tensor([1 / 2, 1 / 12], device=device),
        )


class TestFullMetricTensor:
    def test_non_square_matrices_raise(self, device: torch.device) -> None:
        with pytest.raises(ValueError, match="square"):
            FullMetricTensor(torch.ones(4, 2, 3, device=device))

    def test_non_diagonal_metric(self, device: torch.device) -> None:
        # det(G) = 3 and G^-1 = [[2, -1], [-1, 2]] / 3
        metric = FullMetricTensor(
            torch.tensor([[[2.0, 1.0], [1.0, 2.0]]], device=device)
        )
        v = torch.tensor([[1.0, 0.0]], device=device)
        s = torch.tensor([1.0], device=device)

        sqrt_3 = 3.0**0.5
        assert torch.allclose(metric.sqrt_det, torch.tensor([sqrt_3], device=device))
        assert torch.allclose(
            metric.inv_metric_times_vec(v),
            torch.tensor([[2 / 3, -1 / 3]], device=device),
        )
        assert torch.allclose(
            metric.sqrt_det_times_vec(v), torch.tensor([[sqrt_3, 0]], device=device)
        )
        assert torch.allclose(
            metric.inv_sqrt_det_times_scalar(s),
            torch.tensor([1 / sqrt_3], device=device),
        )

    def test_matches_diagonal_metric_for_a_diagonal_matrix(
        self, device: torch.device
    ) -> None:
        diag = torch.tensor([[1.0, 4.0], [9.0, 16.0]], device=device)
        full = FullMetricTensor(torch.diag_embed(diag))
        diagonal = DiagonalMetricTensor(diag)
        v = torch.randn(2, 2, device=device)

        assert torch.allclose(full.sqrt_det, diagonal.sqrt_det)
        assert torch.allclose(
            full.inv_metric_times_vec(v), diagonal.inv_metric_times_vec(v)
        )

    def test_scale_factors_are_not_defined(self, device: torch.device) -> None:
        metric = FullMetricTensor(torch.eye(2, device=device).expand(3, 2, 2))
        v = torch.ones(3, 2, device=device)

        with pytest.raises(NotImplementedError):
            metric.scale_factors_times_vec(v)
        with pytest.raises(NotImplementedError):
            metric.inv_scale_factors_times_vec(v)
