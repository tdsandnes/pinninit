"""Unit tests for data_structure/coords.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from pinninit.data_structure.coords import Coords, make_coords_dataclass
from pinninit.geometry.coordinate_system import CartesianCoords, CoordSystem
from pinninit.geometry.domain import Domain

from ..helpers import N_POINTS, make_coords


class TestConstruction:
    def test_cartesian_labels_and_tensors(self, coords_2d: Coords) -> None:
        assert coords_2d.coord_labels == ("x", "y")
        assert coords_2d.geometric_labels == ("x", "y")
        assert coords_2d.internal_labels == ("x", "y")
        assert coords_2d.n_points == N_POINTS
        assert coords_2d.tensor.shape == (N_POINTS, 2)
        assert torch.equal(coords_2d.tensor[:, 0], coords_2d.get_coord("x"))
        assert torch.equal(coords_2d.tensor[:, 1], coords_2d.get_coord("y"))

    def test_time_is_a_coord_but_not_geometric(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        x = torch.linspace(0.1, 0.9, N_POINTS, device=device)
        y = torch.linspace(0.2, 0.8, N_POINTS, device=device)
        t = torch.linspace(0.3, 0.7, N_POINTS, device=device)

        coords = make_coords(cartesian_2d, x=x, y=y, t=t)

        assert coords.coord_labels == ("t", "x", "y")
        assert coords.geometric_labels == ("x", "y")
        assert coords.internal_labels == ("t", "x", "y")
        assert coords.geometric.shape == (N_POINTS, 2)

    def test_internal_coords_are_cartesian(
        self, coords_cylindrical: Coords, coords_spherical: Coords
    ) -> None:
        assert coords_cylindrical.geometric_labels == ("r", "theta")
        assert coords_cylindrical.internal_labels == ("x", "y")
        assert coords_spherical.geometric_labels == ("r", "theta", "phi")
        assert coords_spherical.internal_labels == ("x", "y", "z")
        assert coords_spherical.internal.shape == (N_POINTS, 3)

    def test_labels_coord_system_and_device_are_set_on_the_class(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        coords_class = make_coords_dataclass({"y", "x", "t"}, cartesian_2d, device)

        assert coords_class.coord_labels == ("t", "x", "y")
        assert coords_class.geometric_labels == ("x", "y")
        assert coords_class.internal_labels == ("t", "x", "y")
        assert coords_class.coord_system is cartesian_2d
        assert coords_class.device == device

    def test_class_constants_cannot_be_passed_in(
        self, cartesian_2d: CoordSystem, device: torch.device
    ) -> None:
        coords_class = make_coords_dataclass({"x", "y"}, cartesian_2d, device)
        x = y = torch.zeros(3, device=device)
        with pytest.raises(TypeError, match="unexpected keyword"):
            coords_class(x=x, y=y, coord_labels=("x",))  # type: ignore[arg-type]

    def test_float64_coords(
        self, cylindrical_2d: CoordSystem, device: torch.device
    ) -> None:
        r = torch.linspace(0.5, 2.0, N_POINTS, dtype=torch.float64, device=device)
        theta = torch.linspace(0.1, 1.5, N_POINTS, dtype=torch.float64, device=device)

        coords = make_coords(cylindrical_2d, r=r, theta=theta)

        assert coords.dtype == torch.float64
        assert coords.internal.dtype == torch.float64

    def test_coords_are_frozen(self, coords_2d: Coords) -> None:
        with pytest.raises(FrozenInstanceError):
            coords_2d.x = torch.zeros(N_POINTS)  # type: ignore[attr-defined]

    def test_unknown_coord_label_raises(self, coords_2d: Coords) -> None:
        with pytest.raises(KeyError, match="not found in coords"):
            coords_2d.get_coord("z")

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            (set(), "must not be empty"),
            ({"x"}, "requires geometric axes"),
            # Coords already has an attribute called tensor
            ({"x", "y", "tensor"}, "reserved"),
        ],
    )
    def test_invalid_coord_labels_raise(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
        labels: set[str],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            make_coords_dataclass(labels, cartesian_2d, device)

    def test_coord_system_on_another_device_raises(self, device: torch.device) -> None:
        coord_system = CartesianCoords(
            coord_axes={"x", "y"}, device=torch.device("meta")
        )
        with pytest.raises(ValueError, match="coord_system is on meta"):
            make_coords_dataclass({"x", "y"}, coord_system, device)

    @pytest.mark.parametrize(
        ("x_shape", "x_dtype", "error", "match"),
        [
            ((4,), torch.float32, ValueError, "same length"),
            ((3, 2), torch.float32, ValueError, "must be 1D"),
            ((3,), torch.float64, ValueError, "same dtype"),
            ((3,), torch.long, TypeError, "floating point"),
        ],
    )
    def test_invalid_coords_raise(
        self,
        cartesian_2d: CoordSystem,
        device: torch.device,
        x_shape: tuple[int, ...],
        x_dtype: torch.dtype,
        error: type[Exception],
        match: str,
    ) -> None:
        coords_class = make_coords_dataclass({"x", "y"}, cartesian_2d, device)
        x = torch.zeros(x_shape, dtype=x_dtype, device=device)
        y = torch.zeros(3, dtype=torch.float32, device=device)
        with pytest.raises(error, match=match):
            coords_class(x=x, y=y)


class TestInternalCoords:
    def test_cylindrical_derivatives(self, coords_cylindrical: Coords) -> None:
        r = coords_cylindrical.get_coord("r")
        theta = coords_cylindrical.get_coord("theta")

        # internal = (r cos(theta), r sin(theta))
        internal_sum = coords_cylindrical.internal.sum()
        grad_r, grad_theta = torch.autograd.grad(internal_sum, [r, theta])

        assert torch.allclose(grad_r, torch.cos(theta) + torch.sin(theta))
        assert torch.allclose(grad_theta, r * (torch.cos(theta) - torch.sin(theta)))

    def test_gradients_through_geometric_match_gradients_through_coords(
        self, coords_spherical: Coords
    ) -> None:
        coords = coords_spherical
        r, theta, phi = (coords.get_coord(label) for label in ("r", "theta", "phi"))

        grad_geometric, grad_r, grad_theta, grad_phi = torch.autograd.grad(
            coords.internal.sum(), [coords.geometric, r, theta, phi]
        )

        assert torch.allclose(grad_r, grad_geometric[:, 0])
        assert torch.allclose(grad_theta, grad_geometric[:, 1])
        assert torch.allclose(grad_phi, grad_geometric[:, 2])


class TestBatches:
    def test_get_batch(self, coords_2d: Coords, device: torch.device) -> None:
        idx = torch.tensor([0, 2, 4], device=device)

        batch = coords_2d.get_batch(idx)

        assert type(batch) is type(coords_2d)
        assert batch.n_points == 3
        assert batch.parent is coords_2d
        assert batch.parent_idx is not None
        assert torch.equal(batch.parent_idx, idx)
        assert torch.equal(batch.tensor, coords_2d.tensor[idx].detach())

    def test_batch_coords_are_new_leaf_tensors(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        batch = coords_2d.get_batch(torch.tensor([0, 2, 4], device=device))

        x = batch.get_coord("x")
        assert x.requires_grad
        assert x.grad_fn is None

    def test_batch_internal_coords_match_full_coords(
        self, coords_spherical: Coords, device: torch.device
    ) -> None:
        idx = torch.tensor([1, 5, 8], device=device)

        batch = coords_spherical.get_batch(idx)

        assert torch.allclose(batch.internal, coords_spherical.internal[idx].detach())

    def test_nested_batch(self, coords_2d: Coords, device: torch.device) -> None:
        idx1 = torch.tensor([0, 2, 4], device=device)
        idx2 = torch.tensor([0, 2], device=device)

        batch = coords_2d.get_batch(idx1).get_batch(idx2)

        assert torch.equal(batch.tensor, coords_2d.tensor[idx1[idx2]].detach())

    def test_batch_gradients_do_not_flow_to_full_coords(
        self, coords_2d: Coords, device: torch.device
    ) -> None:
        batch = coords_2d.get_batch(torch.tensor([0, 1, 2], device=device))

        batch.internal.sum().backward()  # type: ignore[no-untyped-call]

        assert coords_2d.get_coord("x").grad is None

    def test_float_idx_raises(self, coords_2d: Coords, device: torch.device) -> None:
        with pytest.raises(TypeError, match="integer tensor"):
            coords_2d.get_batch(torch.tensor([0.0, 1.0], device=device))

    def test_2d_idx_raises(self, coords_2d: Coords, device: torch.device) -> None:
        with pytest.raises(ValueError, match="1D"):
            coords_2d.get_batch(torch.tensor([[0, 1], [2, 3]], device=device))


class TestDomainValidation:
    def test_points_inside_domain_pass(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        )
        coords_2d.validate_in_domain(domain)

    def test_points_outside_domain_raise(
        self, coords_2d: Coords, cartesian_2d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=cartesian_2d,
            geometric_bounds={"x": (0.2, 0.8), "y": (0.0, 1.0)},
        )
        with pytest.raises(ValueError, match="outside of domain"):
            coords_2d.validate_in_domain(domain)

    def test_domain_with_other_coord_system_raises(
        self, coords_2d: Coords, spherical_3d: CoordSystem
    ) -> None:
        domain = Domain(
            coord_system=spherical_3d,
        )
        with pytest.raises(ValueError, match="does not match"):
            coords_2d.validate_in_domain(domain)
