# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Controller protocol for the robot runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

#: Action type accepted by Robot.send_action().
RobotAction = "np.ndarray"


class Controller(Protocol):
    """Action-selection interface for the robot runtime.

    Any class implementing these methods is a valid controller.
    Controllers must be synchronous and non-blocking.
    """

    def start(self) -> None:
        """Initialize controller resources. Called once before the loop starts."""
        ...

    def update(self, observation: dict[str, Any]) -> np.ndarray:
        """Select the next action given the current observation.

        This method MUST NOT block. Controllers needing slow IO must
        buffer or poll internally.

        Args:
            observation: Observation mapping with conventional keys
                (``state``, ``images``, ``task``, ``timestamp``, etc.).

        Returns:
            Action array accepted by ``Robot.send_action()``.
        """
        ...

    def stop(self) -> None:
        """Release controller resources. Called during shutdown."""
        ...

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        ...

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Optional: pre-warm controller with ``n`` blocking inferences."""
        ...
