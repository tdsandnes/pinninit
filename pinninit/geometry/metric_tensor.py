from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MetricTensor(ABC):
    """Base class for metric tensors."""

    def __init__(self, n_points: int, device: torch.device):
        self.n_points = n_points
        self.device = device

    @property
    @abstractmethod
    def sqrt_det(self) -> torch.Tensor:
        """
        Square root of determinant of metric tensor.

        Returns
        -------
        torch.Tensor
            Shape: (n_points,)
        """
        pass

    @abstractmethod
    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Multiply a vector by the inverse of the metric tensor.

        Parameters
        ----------
        v : torch.Tensor
            Shape: (n_points, dim)
        """
        pass

    @abstractmethod
    def sqrt_det_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Multiply a vector by the square root of the determinant of the metric tensor.

        Parameters
        ----------
        v : torch.Tensor
            Shape: (n_points, dim)
        """
        pass

    @abstractmethod
    def inv_sqrt_det_times_scalar(self, s: torch.Tensor) -> torch.Tensor:
        """Multiply a scalar by the inverse of the square root of the determinant.

        Parameters
        ----------
        s : torch.Tensor
            Shape: (n_points,)
        """
        pass

    @abstractmethod
    def scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Multiply a vector by the coordinate scale factors.

        Parameters
        ----------
        v : torch.Tensor
            Shape: (n_points, dim)
        """
        pass

    @abstractmethod
    def inv_scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Divide a vector by the coordinate scale factors.

        Parameters
        ----------
        v : torch.Tensor
            Shape: (n_points, dim)
        """
        pass


class IdentityMetricTensor(MetricTensor):
    """Identity metric tensor representation."""

    def __init__(self, n_points: int, device: torch.device):
        super().__init__(n_points, device)

    @property
    def sqrt_det(self) -> torch.Tensor:
        return torch.ones(self.n_points, device=self.device)

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def sqrt_det_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def inv_sqrt_det_times_scalar(self, s: torch.Tensor) -> torch.Tensor:
        return s

    def scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def inv_scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v


class DiagonalMetricTensor(MetricTensor):
    """Diagonal metric tensor representation."""

    def __init__(self, diag: torch.Tensor):
        """
        Parameters
        ----------
        diag : torch.Tensor
            Diagonal elements of the metric tensor. Shape: (n_points, dim)
        """
        super().__init__(n_points=diag.shape[0], device=diag.device)
        self.diag = diag
        self._sqrt_diag: torch.Tensor | None = None
        self._sqrt_det: torch.Tensor | None = None

    @property
    def sqrt_diag(self) -> torch.Tensor:
        if self._sqrt_diag is None:
            self._sqrt_diag = torch.sqrt(self.diag)
        return self._sqrt_diag

    @property
    def sqrt_det(self) -> torch.Tensor:
        if self._sqrt_det is None:
            self._sqrt_det = self.sqrt_diag.prod(dim=1)
        return self._sqrt_det

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v / self.diag

    def sqrt_det_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sqrt_det.unsqueeze(1)

    def inv_sqrt_det_times_scalar(self, s: torch.Tensor) -> torch.Tensor:
        return s / self.sqrt_det

    def scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sqrt_diag

    def inv_scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v / self.sqrt_diag


class FullMetricTensor(MetricTensor):
    """Full metric tensor representation."""

    def __init__(self, G: torch.Tensor):
        """
        Parameters
        ----------
        G : torch.Tensor
            Full metric tensor. Shape: (n_points, dim, dim)
        """
        super().__init__(n_points=G.shape[0], device=G.device)
        self.G = G
        self._inv: torch.Tensor | None = None
        self._sqrt_det: torch.Tensor | None = None

        if G.shape[1] != G.shape[2]:
            raise ValueError("Metric tensor must be square matrices (dim x dim).")

    @property
    def inv(self) -> torch.Tensor:
        if self._inv is None:
            self._inv = torch.linalg.inv(self.G)
        return self._inv

    @property
    def sqrt_det(self) -> torch.Tensor:
        if self._sqrt_det is None:
            self._sqrt_det = torch.sqrt(torch.det(self.G))
        return self._sqrt_det

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        # Batch matrix multiplication: (n_points, dim, dim) @ (n_points, dim, 1)
        return torch.bmm(self.inv, v.unsqueeze(-1)).squeeze(-1)

    def sqrt_det_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        return v * self.sqrt_det.unsqueeze(1)

    def inv_sqrt_det_times_scalar(self, s: torch.Tensor) -> torch.Tensor:
        return s / self.sqrt_det

    def scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Scale factor multiplication only defined for orthogonal metrics."
        )

    def inv_scale_factors_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Inverse scale factor multiplication only defined for orthogonal metrics."
        )
