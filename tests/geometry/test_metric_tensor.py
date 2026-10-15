"""
Unit tests for metric_tensor.py
"""

from __future__ import annotations

import pytest
import torch

from pinninit.geometry.metric_tensor import (
    DiagonalMetricTensor,
    FullMetricTensor,
    IdentityMetricTensor,
)


# IdentityMetricTensor
class TestIdentityMetricTensor:
    def test_sqrt_det(self, device: torch.device) -> None:
        n = 5
        metric = IdentityMetricTensor(n, device)

        assert metric.sqrt_det.shape == (n,)
        assert torch.allclose(metric.sqrt_det, torch.ones(n, device=device))

    def test_inv_metric_times_vec(self, device: torch.device) -> None:
        n = 4
        metric = IdentityMetricTensor(n, device)

        v = torch.randn(n, 3, device=device)
        out = metric.inv_metric_times_vec(v)

        assert torch.allclose(out, v)

    def test_sqrt_det_times_vec(self, device: torch.device) -> None:
        n = 4
        metric = IdentityMetricTensor(n, device)

        v = torch.randn(n, 3, device=device)
        out = metric.sqrt_det_times_vec(v)

        assert torch.allclose(out, v)

    def test_inv_sqrt_det_times_scalar(self, device: torch.device) -> None:
        n = 4
        metric = IdentityMetricTensor(n, device)

        s = torch.randn(n, device=device)
        out = metric.inv_sqrt_det_times_scalar(s)

        assert torch.allclose(out, s)

    def test_scale_factors_times_vec(self, device: torch.device) -> None:
        n = 4
        metric = IdentityMetricTensor(n, device)

        v = torch.randn(n, 3, device=device)
        out = metric.scale_factors_times_vec(v)

        assert torch.allclose(out, v)

    def test_inv_scale_factors_times_vec(self, device: torch.device) -> None:
        n = 4
        metric = IdentityMetricTensor(n, device)

        v = torch.randn(n, 3, device=device)
        out = metric.inv_scale_factors_times_vec(v)

        assert torch.allclose(out, v)


# DiagonalMetricTensor
class TestDiagonalMetricTensor:
    def test_sqrt_diag_and_sqrt_det(self, device: torch.device) -> None:
        diag = torch.tensor(
            [
                [1.0, 4.0, 9.0],
                [4.0, 9.0, 16.0],
                [1.0, 1.0, 1.0],
                [2.0, 8.0, 18.0],
                [9.0, 9.0, 9.0],
                [16.0, 25.0, 36.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        expected_sqrt_diag = torch.sqrt(diag)
        expected_sqrt_det = expected_sqrt_diag.prod(dim=1)

        assert torch.allclose(metric.sqrt_diag, expected_sqrt_diag)
        assert torch.allclose(metric.sqrt_det, expected_sqrt_det)

    def test_inv_metric_times_vec(self, device: torch.device) -> None:
        n = 5

        diag = torch.tensor(
            [
                [1.0, 2.0],
                [4.0, 8.0],
                [9.0, 3.0],
                [16.0, 2.0],
                [25.0, 5.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        v = torch.randn(n, 2, device=device)
        out = metric.inv_metric_times_vec(v)

        assert torch.allclose(out, v / diag)

    def test_sqrt_det_times_vec(self, device: torch.device) -> None:
        n = 4

        diag = torch.tensor(
            [
                [1.0, 4.0],
                [4.0, 9.0],
                [9.0, 16.0],
                [16.0, 25.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        v = torch.randn(n, 2, device=device)
        out = metric.sqrt_det_times_vec(v)

        expected = v * metric.sqrt_det.unsqueeze(1)
        assert torch.allclose(out, expected)

    def test_inv_sqrt_det_times_scalar(self, device: torch.device) -> None:
        n = 4

        diag = torch.tensor(
            [
                [1.0, 4.0],
                [4.0, 9.0],
                [9.0, 16.0],
                [16.0, 25.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        s = torch.randn(n, device=device)
        out = metric.inv_sqrt_det_times_scalar(s)

        assert torch.allclose(out, s / metric.sqrt_det)

    def test_scale_factors_times_vec(self, device: torch.device) -> None:
        n = 3

        diag = torch.tensor(
            [
                [1.0, 4.0],
                [9.0, 16.0],
                [25.0, 36.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        v = torch.randn(n, 2, device=device)
        out = metric.scale_factors_times_vec(v)

        assert torch.allclose(out, v * metric.sqrt_diag)

    def test_inv_scale_factors_times_vec(self, device: torch.device) -> None:
        n = 3

        diag = torch.tensor(
            [
                [1.0, 4.0],
                [9.0, 16.0],
                [25.0, 36.0],
            ],
            device=device,
        )

        metric = DiagonalMetricTensor(diag)

        v = torch.randn(n, 2, device=device)
        out = metric.inv_scale_factors_times_vec(v)

        assert torch.allclose(out, v / metric.sqrt_diag)


# FullMetricTensor
class TestFullMetricTensor:
    def test_invalid_shape_raises(self, device: torch.device) -> None:
        G = torch.randn(4, 2, 3, device=device)

        with pytest.raises(ValueError, match="square"):
            FullMetricTensor(G)

    def test_sqrt_det(self, device: torch.device) -> None:
        n = 4
        dim = 3

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        expected = torch.sqrt(torch.det(G))

        assert torch.allclose(metric.sqrt_det, expected)

    def test_inv_metric_times_vec(self, device: torch.device) -> None:
        n = 4
        dim = 3

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        v = torch.randn(n, dim, device=device)

        out = metric.inv_metric_times_vec(v)

        expected = torch.bmm(torch.linalg.inv(G), v.unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(out, expected)

    def test_sqrt_det_times_vec(self, device: torch.device) -> None:
        n = 4
        dim = 3

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        v = torch.randn(n, dim, device=device)

        out = metric.sqrt_det_times_vec(v)

        expected = v * metric.sqrt_det.unsqueeze(1)

        assert torch.allclose(out, expected)

    def test_inv_sqrt_det_times_scalar(self, device: torch.device) -> None:
        n = 4
        dim = 3

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        s = torch.randn(n, device=device)

        out = metric.inv_sqrt_det_times_scalar(s)

        expected = s / metric.sqrt_det

        assert torch.allclose(out, expected)

    def test_scale_factors_times_vec_raises(self, device: torch.device) -> None:
        n = 3
        dim = 2

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        v = torch.randn(n, dim, device=device)

        with pytest.raises(NotImplementedError):
            metric.scale_factors_times_vec(v)

    def test_inv_scale_factors_times_vec_raises(self, device: torch.device) -> None:
        n = 3
        dim = 2

        G = torch.randn(n, dim, dim, device=device)
        G = G @ G.transpose(-1, -2) + 0.5 * torch.eye(dim, device=device)

        metric = FullMetricTensor(G)

        v = torch.randn(n, dim, device=device)

        with pytest.raises(NotImplementedError):
            metric.inv_scale_factors_times_vec(v)
