from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class PointInStep(Enum):
    FIRST = "first"
    LAST = "last"


class PointInStepRecorder:
    def __init__(self, policy: PointInStep):
        self._policy = policy
        self._value: Any | None = None
        self._has_value: bool = False

    def reset(self) -> None:
        self._value = None
        self._has_value = False

    def record(self, value: Any) -> None:
        if self._policy is PointInStep.FIRST:
            if self._has_value:
                return
            self._value = value
            self._has_value = True

        elif self._policy is PointInStep.LAST:
            self._value = value
            self._has_value = True

        else:
            raise RuntimeError(f"Policy {self._policy} not recognised.")

    @property
    def value(self) -> Any:
        if not self._has_value:
            raise RuntimeError("No value recorded for this step.")
        return self._value

    @property
    def has_value(self) -> bool:
        return self._has_value
