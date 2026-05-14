# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fallback action providers for PolicyController.

A fallback supplies a safe action when the policy cannot produce one,
e.g. during async-inference bootstrap (queue empty, first chunk in flight)
or after a worker exception that drained the queue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class FallbackAction(Protocol):
    """Supplies a safe action when the policy has none.

    Implementations must be pure and fast — they run on the runtime tick.
    """

    def action(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a safe action for the current observation.

        Args:
            observation: Current observation mapping.

        Returns:
            Action array compatible with the robot.
        """
        ...


class HoldStateFallback:
    """Returns ``observation['state']`` as the action.

    Suitable for position-controlled robots: "stay where you are".
    Requires the observation to contain a ``state`` key with a 1-D array
    matching the robot's action dimension.
    """

    def action(self, observation: dict[str, Any]) -> np.ndarray:
        state = observation.get("state")
        if state is None:
            msg = "HoldStateFallback requires observation['state']"
            raise KeyError(msg)
        return state.copy()
