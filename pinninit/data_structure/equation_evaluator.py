from __future__ import annotations

from typing import TYPE_CHECKING

from ..differential_equations.operators import Operators

if TYPE_CHECKING:
    from .coords_dataclass import Coords
    from .data_container import DataContainer
    from .equation import Equation


class EquationEvaluator:
    def __init__(self, data_container: DataContainer) -> None:
        self.data_container = data_container
        self._sorted_equations = self._topological_sort()

    def _visit(
        self,
        equation: Equation,
        output_to_eq: dict[str, Equation],
        visiting: set[str],
        visited: set[str],
        sorted_equations: list[Equation],
    ) -> None:
        """Recursively process an equation after its dependencies."""
        # Already processed.
        if equation.label in visited:
            return

        # Currently being explored.
        if equation.label in visiting:
            raise ValueError(
                f"Cyclic equation dependency involving '{equation.label}'."
            )

        visiting.add(equation.label)

        # Visit all equation dependencies first.
        for input_label in equation.input_labels:
            dep_eq = output_to_eq.get(input_label)
            if dep_eq is not None:
                self._visit(
                    dep_eq,
                    output_to_eq,
                    visiting,
                    visited,
                    sorted_equations,
                )

        visiting.remove(equation.label)
        visited.add(equation.label)

        # Dependencies have been processed, so this equation can be added.
        sorted_equations.append(equation)

    def _topological_sort(self) -> list[Equation]:
        """Return equations ordered so each one appears after anything it depends on."""
        # Map output labels to the equation that produces them.
        output_to_eq: dict[str, Equation] = {}
        for equation in self.data_container.equations:
            for label in equation.output_label_to_rank:
                output_to_eq[label] = equation

        sorted_equations: list[Equation] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        # Visit every equation because the graph may contain
        # multiple disconnected components.
        for equation in self.data_container.equations:
            self._visit(
                equation,
                output_to_eq,
                visiting,
                visited,
                sorted_equations,
            )

        return sorted_equations

    def evaluate(self, coords: Coords) -> None:
        """Evaluate all equations in dependency order."""
        # Set operators
        operators = Operators(coords)

        for equation in self._sorted_equations:
            result = equation.evaluate(self.data_container, coords, operators)

            for label, tensor in result.items():
                self.data_container.set_field_from_tensor(
                    label, tensor, coords, allow_derived=True
                )
