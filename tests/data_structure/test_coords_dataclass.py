"""
Unit tests for coords_dataclass / make_coords_dataclass.
"""

from __future__ import annotations

import numpy as np
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
    make_xyz_3d,
)


# Local fixtures
@pytest.fixture
def device():
    return make_device()


@pytest.fixture
def n_points():
    return N_POINTS


@pytest.fixture
def cartesian_1d(device):
    return make_cartesian_1d(device)


@pytest.fixture
def cartesian_2d(device):
    return make_cartesian_2d(device)


@pytest.fixture
def cartesian_3d(device):
    return make_cartesian_3d(device)


@pytest.fixture
def cylindrical_2d(device):
    return make_cylindrical_2d(device)


@pytest.fixture
def spherical_3d(device):
    return make_spherical_3d(device)


@pytest.fixture
def domain_2d_cartesian(device):
    return make_domain_2d_cartesian(make_cartesian_2d(device), device)


@pytest.fixture
def x_1d(device):
    return make_x_1d(device)


@pytest.fixture
def xy_2d(device):
    return make_xy_2d(device)


@pytest.fixture
def xyz_3d(device):
    return make_xyz_3d(device)


@pytest.fixture
def r_theta_2d(device):
    return make_r_theta_2d(device)


@pytest.fixture
def r_theta_phi_3d(device):
    return make_r_theta_phi_3d(device)


# Helpers
def make_coords_cartesian_1d(coord_system, x, domain=None):
    Coords = make_coords_dataclass({"x"}, x.device)
    return Coords(x=x, coord_system=coord_system, domain=domain)


def make_coords_cartesian_2d(coord_system, x, y, domain=None):
    Coords = make_coords_dataclass({"x", "y"}, x.device)
    return Coords(x=x, y=y, coord_system=coord_system, domain=domain)


def make_coords_cartesian_3d(coord_system, x, y, z, domain=None):
    Coords = make_coords_dataclass({"x", "y", "z"}, x.device)
    return Coords(x=x, y=y, z=z, coord_system=coord_system, domain=domain)


def make_coords_cylindrical_2d(coord_system, r, theta, domain=None):
    Coords = make_coords_dataclass({"r", "theta"}, r.device)
    return Coords(
        r=r,
        theta=theta,
        coord_system=coord_system,
        domain=domain,
    )


def make_coords_spherical_3d(coord_system, r, theta, phi, domain=None):
    Coords = make_coords_dataclass({"r", "theta", "phi"}, r.device)
    return Coords(
        r=r,
        theta=theta,
        phi=phi,
        coord_system=coord_system,
        domain=domain,
    )


# Construction and basic attributes
class TestCoordsConstructionCartesian:
    def test_coordinate_metadata_consistency(self, cartesian_2d, xy_2d):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        assert set(coords.coord_labels) == {"x", "y"}
        for i, label in enumerate(coords.coord_labels):
            assert coords.label_to_index[label] == i
        assert set(coords.geometric_labels) == set(cartesian_2d.axes_labels_selected)

    def test_tensor_columns_match_coord_labels(
        self,
        cartesian_2d,
        xy_2d,
    ):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        for i, label in enumerate(coords.coord_labels):
            assert torch.allclose(
                coords.tensor[:, i],
                getattr(coords, label),
            )

    def test_all_coord_tensors_have_requires_grad(
        self,
        cartesian_2d,
        xy_2d,
    ):
        x, y = xy_2d
        x.requires_grad_(True)
        y.requires_grad_(True)

        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        for label in coords.coord_labels:
            assert getattr(coords, label).requires_grad is True
        assert coords.tensor.requires_grad is True

    def test_n_points_correct(self, cartesian_2d, xy_2d, n_points):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        assert coords.n_points == n_points


# Cartesian: internal coords are identity map of geometric coords
class TestCartesianInternalCoords:
    def test_internal_labels_match_geometric_labels(
        self,
        cartesian_2d,
        xy_2d,
    ):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        assert coords.internal_labels == coords.geometric_labels

    def test_diff_internal_wrt_coords(
        self,
        cartesian_2d,
        xy_2d,
    ):
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
        cartesian_2d,
        xy_2d,
    ):
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
        cylindrical_2d,
        r_theta_2d,
    ):
        r, theta = r_theta_2d

        coords = make_coords_cylindrical_2d(
            cylindrical_2d,
            r,
            theta,
        )

        assert coords.internal_labels == ("x", "y")

    def test_diff_internal_wrt_coords(
        self,
        cylindrical_2d,
        r_theta_2d,
    ):
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
        cylindrical_2d,
        r_theta_2d,
    ):
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
        spherical_3d,
        r_theta_phi_3d,
    ):
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
        spherical_3d,
        r_theta_phi_3d,
    ):
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
        spherical_3d,
        r_theta_phi_3d,
    ):
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
        cartesian_1d,
        device,
        n_points,
    ):
        x = torch.linspace(0.1, 0.9, n_points, device=device, requires_grad=True)
        t = torch.linspace(0.0, 1.0, n_points, device=device, requires_grad=True)

        Coords = make_coords_dataclass({"x", "t"}, device)

        coords = Coords(
            x=x,
            t=t,
            coord_system=cartesian_1d,
            domain=None,
        )

        assert "t" in coords.coord_labels
        assert "t" not in coords.geometric_labels
        assert "t" in coords.internal_labels
        assert coords.tensor.shape == (n_points, 2)
        assert coords.geometric.shape == (n_points, 1)
        assert coords.internal.shape == (n_points, 2)

    def test_diff_internal_wrt_coords(
        self,
        cartesian_2d,
        device,
        n_points,
    ):
        x = torch.linspace(0.1, 0.9, n_points, device=device, requires_grad=True)
        y = torch.linspace(0.2, 0.8, n_points, device=device, requires_grad=True)
        t = torch.linspace(0.0, 1.0, n_points, device=device, requires_grad=True)

        Coords = make_coords_dataclass({"x", "y", "t"}, device)

        coords = Coords(
            x=x,
            y=y,
            t=t,
            coord_system=cartesian_2d,
            domain=None,
        )

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
        cartesian_2d,
        device,
        n_points,
    ):
        x = torch.linspace(0.1, 0.9, n_points, device=device, requires_grad=True)
        y = torch.linspace(0.2, 0.8, n_points, device=device, requires_grad=True)
        t = torch.linspace(0.0, 1.0, n_points, device=device, requires_grad=True)

        Coords = make_coords_dataclass({"x", "y", "t"}, device)

        coords = Coords(
            x=x,
            y=y,
            t=t,
            coord_system=cartesian_2d,
            domain=None,
        )

        loss = coords.internal.sum()

        grad = torch.autograd.grad(
            outputs=loss,
            inputs=coords.geometric,
            retain_graph=False,
            allow_unused=False,
        )[0]

        assert grad.shape == coords.geometric.shape
        assert torch.allclose(grad, torch.ones_like(coords.geometric))


# internal_lb and internal_ub
class TestInternalBounds:
    def test_internal_lb_ub(
        self,
        cartesian_2d,
        xy_2d,
    ):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        expected_lb = torch.min(
            coords.internal.detach(),
            dim=0,
        ).values

        expected_ub = torch.max(
            coords.internal,
            dim=0,
        ).values

        assert torch.allclose(
            coords.internal_lb,
            expected_lb,
        )

        assert torch.allclose(
            coords.internal_ub.detach(),
            expected_ub,
        )
        

# Domain validation
class TestDomainValidation:
    def test_points_inside_domain_pass(
        self,
        cartesian_2d,
        xy_2d,
        domain_2d_cartesian,
    ):
        x, y = xy_2d

        assert (
            make_coords_cartesian_2d(
                cartesian_2d,
                x,
                y,
                domain_2d_cartesian,
            )
            is not None
        )

    def test_points_outside_domain_raise(
        self,
        cartesian_2d,
        device,
    ):
        x = torch.tensor([0.5, 1.5, 0.3], device=device, requires_grad=True)
        y = torch.tensor([0.5, 0.5, 0.5], device=device, requires_grad=True)

        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={
                "x": (0.0, 1.0),
                "y": (0.0, 1.0),
            },
            periodic_coords=None,
            device=device,
        )

        Coords = make_coords_dataclass({"x", "y"}, device)

        with pytest.raises(ValueError, match="outside of domain"):
            Coords(
                x=x,
                y=y,
                coord_system=cartesian_2d,
                domain=domain,
            )


class TestGetBatch:
    def test_get_batch_returns_coords_protocol(self, cartesian_2d, xy_2d):
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
        assert torch.allclose(batch_coords.internal_lb, coords.internal_lb)
        assert torch.allclose(batch_coords.internal_ub, coords.internal_ub)
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
        self, cylindrical_2d, r_theta_2d
    ):
        r, theta = r_theta_2d
        coords = make_coords_cylindrical_2d(cylindrical_2d, r, theta)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)

        expected_x = r[idx] * torch.cos(theta[idx])
        expected_y = r[idx] * torch.sin(theta[idx])

        x_idx = coords.internal_label_to_index["x"]
        y_idx = coords.internal_label_to_index["y"]

        assert torch.allclose(batch_coords.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch_coords.internal[:, y_idx], expected_y.detach())

    def test_get_batch_internal_values_correct_spherical(
        self, spherical_3d, r_theta_phi_3d
    ):
        r, theta, phi = r_theta_phi_3d
        coords = make_coords_spherical_3d(spherical_3d, r, theta, phi)

        idx = torch.tensor([0, 2, 4])
        batch_coords = coords.get_batch(idx)

        expected_x = r[idx] * torch.sin(theta[idx]) * torch.cos(phi[idx])
        expected_y = r[idx] * torch.sin(theta[idx]) * torch.sin(phi[idx])
        expected_z = r[idx] * torch.cos(theta[idx])

        x_idx = coords.internal_label_to_index["x"]
        y_idx = coords.internal_label_to_index["y"]
        z_idx = coords.internal_label_to_index["z"]

        assert torch.allclose(batch_coords.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch_coords.internal[:, y_idx], expected_y.detach())
        assert torch.allclose(batch_coords.internal[:, z_idx], expected_z.detach())

    def test_batch_grad_does_not_flow_to_parent_coords(self, cartesian_2d, xy_2d):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)
        batch = coords.get_batch(torch.tensor([0, 1, 2]))

        loss = batch.internal.sum()
        loss.backward()

        assert x.grad is None
        assert y.grad is None

    def test_nested_batch_internal_values_correct(self, cartesian_2d, xy_2d):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx1 = torch.tensor([0, 2, 4])
        idx2 = torch.tensor([0, 2])
        batch1 = coords.get_batch(idx1)
        batch2 = batch1.get_batch(idx2)

        # idx2 into idx1 maps to root indices [0, 4]
        expected = coords.internal[idx1[idx2]].detach()
        assert torch.allclose(batch2.internal, expected)

    def test_nested_batch_grad_wrt_batch_coords(self, cartesian_2d, xy_2d):
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

    def test_get_batch_float_idx_raises(self, cartesian_2d, xy_2d):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        float_idx = torch.tensor([0.0, 1.0, 2.0])
        with pytest.raises(TypeError, match="integer tensor"):
            coords.get_batch(float_idx)

    def test_get_batch_2d_idx_raises(self, cartesian_2d, xy_2d):
        x, y = xy_2d
        coords = make_coords_cartesian_2d(cartesian_2d, x, y)

        idx_2d = torch.tensor([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="1D"):
            coords.get_batch(idx_2d)

    def test_diff_batch_internal_wrt_batch_coords_cartesian(self, cartesian_2d, xy_2d):
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
        self, cartesian_2d, xy_2d
    ):
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

    def test_nested_batch_internal_values_correct_spherical(self, spherical_3d, r_theta_phi_3d):
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

        x_idx = coords.internal_label_to_index["x"]
        y_idx = coords.internal_label_to_index["y"]
        z_idx = coords.internal_label_to_index["z"]

        assert torch.allclose(batch2.internal[:, x_idx], expected_x.detach())
        assert torch.allclose(batch2.internal[:, y_idx], expected_y.detach())
        assert torch.allclose(batch2.internal[:, z_idx], expected_z.detach())


# Validation errors
class TestCoordsValidationErrors:
    @pytest.mark.parametrize(
        "invalid_input",
        [
            [0.1, 0.2, 0.3],  # list
            (0.1, 0.2, 0.3),  # tuple
            np.array([0.1, 0.2, 0.3]),  # numpy array
        ],
    )
    def test_invalid_types_raise(self, cartesian_2d, device, invalid_input):
        y = torch.linspace(0.0, 1.0, 3, device=device, requires_grad=True)
        Coords = make_coords_dataclass({"x", "y"}, device)

        with pytest.raises(TypeError, match="torch.Tensor"):
            Coords(x=invalid_input, y=y, coord_system=cartesian_2d, domain=None)

    def test_mismatched_coord_lengths_raise(
        self,
        cartesian_2d,
        device,
    ):
        x = torch.linspace(0.0, 1.0, 10, device=device, requires_grad=True)
        y = torch.linspace(0.0, 1.0, 7, device=device, requires_grad=True)

        Coords = make_coords_dataclass({"x", "y"}, device)

        with pytest.raises(ValueError, match="same length"):
            Coords(
                x=x,
                y=y,
                coord_system=cartesian_2d,
                domain=None,
            )

    def test_2d_coord_tensor_raises(
        self,
        cartesian_1d,
        device,
    ):
        x = torch.ones((5, 2), device=device, requires_grad=True)

        Coords = make_coords_dataclass({"x"}, device)

        with pytest.raises(ValueError, match="1D"):
            Coords(
                x=x,
                coord_system=cartesian_1d,
                domain=None,
            )

    def test_empty_coordinates_raises(self, cartesian_1d, device):
        # Tests the 'if not lengths:' block
        Coords = make_coords_dataclass(set(), device)
        with pytest.raises(ValueError, match="No coordinates provided"):
            Coords(coord_system=cartesian_1d, domain=None)

    def test_duplicate_labels_in_factory_raises(self, device):
        """Verify that passing duplicate labels to the function raises a ValueError."""
        duplicate_list = ["x", "y", "x"]  # 'x' is repeated

        with pytest.raises(ValueError, match="Duplicate coordinate labels provided"):
            make_coords_dataclass(duplicate_list, device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mismatched_devices_raises(self, cartesian_2d):
        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")

        # Build clean tensors matching your suite's requirements
        x = torch.linspace(0.0, 1.0, 10, device=cpu_device, requires_grad=True)
        y = torch.linspace(0.0, 1.0, 10, device=gpu_device, requires_grad=True)

        Coords = make_coords_dataclass({"x", "y"}, cpu_device)
        with pytest.raises(ValueError, match="expected cpu"):
            Coords(
                x=x,
                y=y,
                coord_system=cartesian_2d,
                domain=None,
            )

    def test_missing_coord_system_raises(self, device, n_points):
        x = torch.linspace(0.0, 1.0, n_points, device=device, requires_grad=True)
        Coords = make_coords_dataclass({"x"}, device)
        with pytest.raises(
            TypeError, match="coord_system must be a CoordSystem instance"
        ):
            Coords(x=x, coord_system=None, domain=None)
