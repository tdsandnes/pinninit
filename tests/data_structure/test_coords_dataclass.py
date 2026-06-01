"""
Unit tests for coords_dataclass / make_coords_dataclass.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from pinninit.data_structure.coords_dataclass import (
    CoordsProtocol,
    make_coords_dataclass,
)
from pinninit.geometry.domain import Domain

from .fixtures import (
    N_POINTS,
    make_cartesian_1d,
    make_cartesian_2d,
    make_cartesian_3d,
    make_cylindrical_2d,
    make_device,
    make_domain_2d_cartesian,
    make_r_theta_2d,
    make_r_theta_phi_3d,
    make_spherical_3d,
    make_x_1d,
    make_xy_2d,
    make_xyt_2d,
    make_xyz_3d,
)


# Local fixtures
@pytest.fixture  # type: ignore[misc]
def device() -> torch.device:
    return make_device()


@pytest.fixture  # type: ignore[misc]
def n_points() -> int:
    return N_POINTS


@pytest.fixture  # type: ignore[misc]
def cartesian_1d(device: torch.device) -> Any:
    return make_cartesian_1d(device)


@pytest.fixture  # type: ignore[misc]
def cartesian_2d(device: torch.device) -> Any:
    return make_cartesian_2d(device)


@pytest.fixture  # type: ignore[misc]
def cartesian_3d(device: torch.device) -> Any:
    return make_cartesian_3d(device)


@pytest.fixture  # type: ignore[misc]
def cylindrical_2d(device: torch.device) -> Any:
    return make_cylindrical_2d(device)


@pytest.fixture  # type: ignore[misc]
def spherical_3d(device: torch.device) -> Any:
    return make_spherical_3d(device)


@pytest.fixture  # type: ignore[misc]
def domain_2d_cartesian(device: torch.device, cartesian_2d: Any) -> Domain:
    return make_domain_2d_cartesian(cartesian_2d, device)


@pytest.fixture  # type: ignore[misc]
def x_1d(device: torch.device) -> torch.Tensor:
    return make_x_1d(device)


@pytest.fixture  # type: ignore[misc]
def xy_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_xy_2d(device)


@pytest.fixture  # type: ignore[misc]
def xyt_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return make_xyt_2d(device)


@pytest.fixture  # type: ignore[misc]
def xyz_3d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return make_xyz_3d(device)


@pytest.fixture  # type: ignore[misc]
def r_theta_2d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return make_r_theta_2d(device)


@pytest.fixture  # type: ignore[misc]
def r_theta_phi_3d(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return make_r_theta_phi_3d(device)


# Helpers
def make_coords_cartesian_1d(coord_system: Any, x: torch.Tensor) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"x"}, coord_system=coord_system, device=x.device
    )
    return Coords(x=x)


def make_coords_cartesian_2d(
    coord_system: Any, x: torch.Tensor, y: torch.Tensor
) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"x", "y"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y)


def make_coords_cartesian_2d_with_time(
    coord_system: Any, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"x", "y", "t"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y, t=t)


def make_coords_cartesian_3d(
    coord_system: Any, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"x", "y", "z"},
        coord_system=coord_system,
        device=x.device,
    )
    return Coords(x=x, y=y, z=z)


def make_coords_cylindrical_2d(
    coord_system: Any, r: torch.Tensor, theta: torch.Tensor
) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"r", "theta"},
        coord_system=coord_system,
        device=r.device,
    )
    return Coords(r=r, theta=theta)


def make_coords_spherical_3d(
    coord_system: Any, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
) -> Any:
    Coords: Any = make_coords_dataclass(
        coord_labels={"r", "theta", "phi"},
        coord_system=coord_system,
        device=r.device,
    )
    return Coords(r=r, theta=theta, phi=phi)


# Construction and basic attributes
class TestCoordsConstructionCartesian:
    def test_coordinate_metadata_consistency(
        self,
        cartesian_3d: Any,
        xyz_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y, z = xyz_3d
        r, theta, phi = r_theta_phi_3d
        cartesian_coords = make_coords_cartesian_3d(cartesian_3d, x, y, z)
        spherical_coords = make_coords_spherical_3d(spherical_3d, r, theta, phi)

        assert cartesian_coords.coord_labels == tuple(sorted(("z", "y", "x")))
        assert cartesian_coords.geometric_labels == cartesian_3d.axes_labels_selected
        assert cartesian_coords.internal_labels == tuple(sorted(("z", "y", "x")))
        assert spherical_coords.coord_labels == tuple(sorted(("theta", "r", "phi")))
        assert spherical_coords.geometric_labels == spherical_3d.axes_labels_selected
        assert spherical_coords.internal_labels == tuple(sorted(("z", "y", "x")))

    def test_tensor_columns_match_coord_labels(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        for i, label in enumerate(coords.coord_labels):
            assert torch.allclose(
                coords.tensor[:, i],
                getattr(coords, label),
            )

    def test_all_coord_tensors_have_requires_grad(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        x.requires_grad_(True)
        y.requires_grad_(True)

        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        for label in coords.coord_labels:
            assert getattr(coords, label).requires_grad is True
        assert coords.tensor.requires_grad is True

    def test_n_points_correct(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        n_points: int,
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        assert coords.n_points == n_points

    def test_coords_are_immutable(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        with pytest.raises(FrozenInstanceError):
            coords.x = torch.zeros_like(coords.x)

        with pytest.raises(FrozenInstanceError):
            coords.n_points = 123


# Cartesian: internal coords are identity map of geometric coords
class TestCartesianInternalCoords:
    def test_internal_labels_match_geometric_labels(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        assert coords.internal_labels == coords.geometric_labels

    def test_diff_internal_wrt_coords(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d

        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        loss = coords.internal.sum()

        inputs = (coords.x, coords.y)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=inputs,
            retain_graph=False,
            allow_unused=False,
        )

        for grad, input in zip(grads, inputs, strict=True):
            assert grad.shape == input.shape
            assert torch.allclose(grad, torch.ones_like(input))

    def test_diff_internal_wrt_geometric(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d

        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert grad.shape == coords.geometric.shape
        assert torch.allclose(grad, torch.ones_like(coords.geometric))


# Cylindrical: internal coords are Cartesian (x, y from r, theta)
class TestCylindricalInternalCoords:
    def test_internal_labels_are_cartesian(
        self,
        cylindrical_2d: Any,
        r_theta_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta = r_theta_2d

        coords = make_coords_cylindrical_2d(
            cylindrical_2d,
            r,
            theta,
        )

        assert coords.internal_labels == ("x", "y")

    def test_diff_internal_wrt_coords(
        self,
        cylindrical_2d: Any,
        r_theta_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta = r_theta_2d

        coords = make_coords_cylindrical_2d(
            cylindrical_2d,
            r,
            theta,
        )

        loss = coords.internal.sum()

        inputs = (coords.r, coords.theta)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=inputs,
            retain_graph=False,
            allow_unused=False,
        )

        expected_r = torch.cos(theta) + torch.sin(theta)
        expected_theta = r * (-torch.sin(theta) + torch.cos(theta))
        expecteds = (expected_r, expected_theta)

        for grad, input, expected in zip(grads, inputs, expecteds, strict=True):
            assert grad.shape == input.shape
            assert torch.allclose(grad, expected)

    def test_diff_internal_wrt_geometric(
        self,
        cylindrical_2d: Any,
        r_theta_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta = r_theta_2d

        coords = make_coords_cylindrical_2d(
            cylindrical_2d,
            r,
            theta,
        )

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        expected_r = torch.cos(theta) + torch.sin(theta)
        expected_theta = r * (-torch.sin(theta) + torch.cos(theta))
        expected = torch.stack((expected_r, expected_theta), dim=1)

        assert grad.shape == coords.geometric.shape
        assert torch.allclose(grad, expected)


# Spherical: internal coords are Cartesian (x, y, z from r, theta, phi)
class TestSphericalInternalCoords:
    def test_internal_labels_are_cartesian(
        self,
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta, phi = r_theta_phi_3d

        coords = make_coords_spherical_3d(
            spherical_3d,
            r,
            theta,
            phi,
        )

        assert coords.internal_labels == ("x", "y", "z")

    def test_diff_internal_wrt_coords(
        self,
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta, phi = r_theta_phi_3d

        coords = make_coords_spherical_3d(
            spherical_3d,
            r,
            theta,
            phi,
        )

        loss = coords.internal.sum()

        inputs = (coords.r, coords.theta, coords.phi)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=inputs,
            retain_graph=False,
            allow_unused=False,
        )

        expected_r = (
            torch.sin(theta) * torch.cos(phi)
            + torch.sin(theta) * torch.sin(phi)
            + torch.cos(theta)
        )

        expected_theta = r * (
            torch.cos(theta) * (torch.cos(phi) + torch.sin(phi)) - torch.sin(theta)
        )

        expected_phi = r * torch.sin(theta) * (-torch.sin(phi) + torch.cos(phi))

        expecteds = (expected_r, expected_theta, expected_phi)

        for grad, input, expected in zip(grads, inputs, expecteds, strict=True):
            assert grad.shape == input.shape
            assert torch.allclose(grad, expected)

    def test_diff_internal_wrt_geometric(
        self,
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta, phi = r_theta_phi_3d

        coords = make_coords_spherical_3d(
            spherical_3d,
            r,
            theta,
            phi,
        )

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        expected_r = (
            torch.sin(theta) * torch.cos(phi)
            + torch.sin(theta) * torch.sin(phi)
            + torch.cos(theta)
        )

        expected_theta = r * (
            torch.cos(theta) * (torch.cos(phi) + torch.sin(phi)) - torch.sin(theta)
        )

        expected_phi = r * torch.sin(theta) * (-torch.sin(phi) + torch.cos(phi))

        expected = torch.stack(
            (expected_r, expected_theta, expected_phi),
            dim=1,
        )

        assert grad.shape == coords.geometric.shape
        assert torch.allclose(grad, expected)


# Non-geometric extra coordinate (e.g. time)
class TestNonGeometricCoord:
    def test_time_coord_in_coord_labels(
        self,
        cartesian_2d: Any,
        xyt_2d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        n_points: int,
    ) -> None:
        x, y, t = xyt_2d

        coords = make_coords_cartesian_2d_with_time(cartesian_2d, x, y, t)

        assert "t" in coords.coord_labels
        assert "t" not in coords.geometric_labels
        assert "t" in coords.internal_labels
        assert coords.tensor.shape == (n_points, 3)
        assert coords.geometric.shape == (n_points, 2)
        assert coords.internal.shape == (n_points, 3)

    def test_diff_internal_wrt_coords(
        self,
        cartesian_2d: Any,
        xyt_2d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y, t = xyt_2d

        coords = make_coords_cartesian_2d_with_time(cartesian_2d, x, y, t)

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.t,
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert grad.shape == coords.t.shape
        assert torch.allclose(grad, torch.ones_like(coords.t))

    def test_diff_internal_wrt_geometric(
        self,
        cartesian_2d: Any,
        xyt_2d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        x, y, t = xyt_2d

        coords = make_coords_cartesian_2d_with_time(cartesian_2d, x, y, t)

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert grad.shape == coords.geometric.shape
        assert torch.allclose(grad, torch.ones_like(coords.geometric))


# Domain validation
class TestDomainValidation:
    def test_points_inside_domain_pass(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        domain_2d_cartesian: Domain,
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        coords.validate_in_domain(domain_2d_cartesian)

    def test_points_outside_domain_raise(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.2, 0.8), "y": (0.0, 1.0)},
            periodic_coords=None,
            device=device,
        )
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        with pytest.raises(ValueError, match="outside of domain"):
            coords.validate_in_domain(domain)

    def test_mismatched_coord_system_raises(
        self,
        cartesian_2d: Any,
        spherical_3d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        domain = Domain(
            coord_system=spherical_3d,
            geometric_bounds=None,
            periodic_coords=None,
            device=device,
        )
        with pytest.raises(ValueError, match="Domain"):
            coords.validate_in_domain(domain)


class TestGetBatch:
    def test_get_batch_returns_coords_protocol(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        assert coords.is_batch is False

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)

        required = CoordsProtocol.__annotations__.keys()
        assert all(hasattr(batch_coords, key) for key in required)
        assert batch_coords.n_points == len(idx)
        assert batch_coords.is_batch is True
        assert batch_coords.parent is coords
        assert batch_coords.coord_labels == coords.coord_labels
        assert batch_coords.internal_labels == coords.internal_labels
        assert batch_coords.geometric_labels == coords.geometric_labels
        assert batch_coords.parent is coords
        assert batch_coords.__class__ is coords.__class__

        # Batch coord tensors have requires_grad True
        for label in batch_coords.coord_labels:
            assert getattr(batch_coords, label).requires_grad is True

        # Detaches from original graph
        for label in batch_coords.coord_labels:
            assert getattr(batch_coords, label).grad_fn is None

    def test_get_batch_internal_values_correct_cylindrical(
        self,
        cylindrical_2d: Any,
        r_theta_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta = r_theta_2d
        coords = make_coords_cylindrical_2d(cylindrical_2d, r, theta)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)

        expected_x = r[idx] * torch.cos(theta[idx])
        expected_y = r[idx] * torch.sin(theta[idx])

        internal_label_to_index = {
            label: i for i, label in enumerate(coords.internal_labels)
        }
        x_idx = internal_label_to_index["x"]
        y_idx = internal_label_to_index["y"]

        assert torch.allclose(batch_coords.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch_coords.internal[:, y_idx], expected_y.detach())

    def test_get_batch_internal_values_correct_spherical(
        self,
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta, phi = r_theta_phi_3d
        coords = make_coords_spherical_3d(spherical_3d, r, theta, phi)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)

        expected_x = r[idx] * torch.sin(theta[idx]) * torch.cos(phi[idx])
        expected_y = r[idx] * torch.sin(theta[idx]) * torch.sin(phi[idx])
        expected_z = r[idx] * torch.cos(theta[idx])

        internal_label_to_index = {
            label: i for i, label in enumerate(coords.internal_labels)
        }
        x_idx = internal_label_to_index["x"]
        y_idx = internal_label_to_index["y"]
        z_idx = internal_label_to_index["z"]

        assert torch.allclose(batch_coords.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch_coords.internal[:, y_idx], expected_y.detach())
        assert torch.allclose(batch_coords.internal[:, z_idx], expected_z.detach())

    def test_batch_grad_does_not_flow_to_parent_coords(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        batch = coords.get_batch(torch.tensor([0, 1, 2]))

        loss = batch.internal.sum()
        loss.backward()

        assert x.grad is None
        assert y.grad is None

    def test_nested_batch_internal_values_correct(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx1 = torch.tensor([0, 2, 4])
        idx2 = torch.tensor([0, 2])
        batch1 = coords.get_batch(idx1)
        batch2 = batch1.get_batch(idx2)

        # idx2 into idx1 maps to root indices [0, 4]
        expected = coords.internal[idx1[idx2]].detach()
        assert torch.allclose(batch2.internal, expected)

    def test_nested_batch_grad_wrt_batch_coords(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx1 = torch.tensor([0, 2, 4])
        idx2 = torch.tensor([0, 2])
        batch2 = coords.get_batch(idx1).get_batch(idx2)

        loss = batch2.internal.sum()
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=(batch2.x, batch2.y),
            retain_graph=False,
            allow_unused=False,
        )

        for grad, coord in zip(grads, (batch2.x, batch2.y), strict=True):
            assert grad.shape == coord.shape
            assert torch.allclose(grad, torch.ones_like(coord))

    def test_get_batch_float_idx_raises(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        float_idx = torch.tensor([0.0, 1.0, 2.0])
        with pytest.raises(TypeError, match="integer tensor"):
            coords.get_batch(float_idx)

    def test_get_batch_2d_idx_raises(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx_2d = torch.tensor([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="1D"):
            coords.get_batch(idx_2d)

    def test_diff_batch_internal_wrt_batch_coords_cartesian(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)
        loss = batch_coords.internal.sum()

        grads = torch.autograd.grad(
            outputs=loss,
            inputs=(batch_coords.x, batch_coords.y),
            retain_graph=False,
            allow_unused=False,
        )

        for grad, coord in zip(grads, (batch_coords.x, batch_coords.y), strict=True):
            assert grad.shape == coord.shape
            assert torch.allclose(grad, torch.ones_like(coord))

    def test_diff_batch_internal_wrt_batch_geometric_cartesian(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)
        loss = batch_coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=batch_coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert grad.shape == batch_coords.geometric.shape
        assert torch.allclose(grad, torch.ones_like(batch_coords.geometric))

    def test_nested_batch_internal_values_correct_spherical(
        self,
        spherical_3d: Any,
        r_theta_phi_3d: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        r, theta, phi = r_theta_phi_3d
        coords = make_coords_spherical_3d(spherical_3d, r, theta, phi)
        idx1 = torch.tensor([0, 2, 4])
        idx2 = torch.tensor([0, 2])
        batch2 = coords.get_batch(idx1).get_batch(idx2)

        # Manually compute absolute expected positions from absolute root indexes [0, 4]
        root_idx = idx1[idx2]
        expected_x = r[root_idx] * torch.sin(theta[root_idx]) * torch.cos(phi[root_idx])
        expected_y = r[root_idx] * torch.sin(theta[root_idx]) * torch.sin(phi[root_idx])
        expected_z = r[root_idx] * torch.cos(theta[root_idx])

        internal_label_to_index = {
            label: i for i, label in enumerate(coords.internal_labels)
        }
        x_idx = internal_label_to_index["x"]
        y_idx = internal_label_to_index["y"]
        z_idx = internal_label_to_index["z"]

        assert torch.allclose(batch2.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch2.internal[:, y_idx], expected_y.detach())
        assert torch.allclose(batch2.internal[:, z_idx], expected_z.detach())


# Validation errors
class TestCoordsValidationErrors:
    def test_mismatched_coord_lengths_raise(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        y = y[:-2]

        with pytest.raises(ValueError, match="same length"):
            make_coords_cartesian_2d(
                cartesian_2d,
                x,
                y,
            )

    def test_2d_coord_tensor_raises(
        self,
        cartesian_1d: Any,
        device: torch.device,
    ) -> None:
        x = torch.ones((5, 2), device=device, requires_grad=True)

        with pytest.raises(ValueError, match="1D"):
            make_coords_cartesian_1d(
                cartesian_1d,
                x,
            )

    def test_empty_coordinates_raises(
        self,
        cartesian_1d: Any,
        device: torch.device,
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            make_coords_dataclass(
                coord_labels=set(), coord_system=cartesian_1d, device=device
            )

    def test_cannot_set_undeclared_attribute(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        with pytest.raises((AttributeError, TypeError)):
            coords.theta = torch.zeros_like(x)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  # type: ignore[misc]
    def test_mismatched_devices_raises(
        self,
        cartesian_2d: Any,
        xy_2d: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, y = xy_2d
        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        x = x.to(cpu_device)
        y = y.to(gpu_device)

        with pytest.raises(ValueError, match="expected cpu"):
            make_coords_cartesian_2d(
                cartesian_2d,
                x,
                y,
            )
