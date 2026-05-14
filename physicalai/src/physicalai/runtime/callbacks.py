# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Runtime callback protocol and base class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class RuntimeCallback:
    """Base callback for runtime side effects.

    Override only the hooks you need. All methods are no-ops by default.
    """

    def on_start(self) -> None:
        """Called once when the runtime loop begins."""

    def on_observation(self, observation: dict[str, Any]) -> None:
        """Called after observation is read and augmented."""

    def before_send_action(self, action: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        """Called before safety and send_action. May modify the action.

        Args:
            action: The action from the controller.
            observation: Current observation.

        Returns:
            Potentially modified action.
        """
        return action

    def on_action_sent(self, action: np.ndarray, observation: dict[str, Any]) -> None:
        """Called after action is sent to the robot."""

    def on_error(self, error: Exception, observation: dict[str, Any] | None = None) -> None:
        """Called when a transient controller error occurs."""

    def on_stop(self) -> None:
        """Called during shutdown."""
