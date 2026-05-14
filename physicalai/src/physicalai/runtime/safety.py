# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Safety layer protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class SafetyViolationError(Exception):
    """Raised when an action cannot be made safe."""


class SafetyLayer(Protocol):
    """Filters actions before they reach the robot."""

    def filter(self, action: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        """Filter or clamp an action.

        Args:
            action: Proposed action.
            observation: Current observation.

        Returns:
            Safe action.

        Raises:
            SafetyViolationError: If the action cannot be made safe.
        """
        ...
