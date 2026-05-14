# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyController — adapts InferenceModel into a Controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from physicalai.runtime.action_queue import ActionQueue

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.model import InferenceModel
    from physicalai.runtime.execution import InferenceExecution


class PolicyController:
    """Adapts InferenceModel + InferenceExecution + ActionQueue into a Controller.

    This is the standard controller for policy deployment. It delegates
    inference scheduling to the execution strategy and pops one action
    per tick from the queue.
    """

    def __init__(
        self,
        model: InferenceModel,
        execution: InferenceExecution,
        action_queue: ActionQueue | None = None,
    ) -> None:
        self._model = model
        self._execution = execution
        self._action_queue = action_queue if action_queue is not None else ActionQueue()
        self._last_action: np.ndarray | None = None

    def start(self) -> None:
        """Start the execution strategy."""
        self._execution.start(self._action_queue, self._model)

    def update(self, observation: dict[str, Any]) -> np.ndarray:
        """Request inference if needed and pop one action.

        Args:
            observation: Current observation mapping.

        Returns:
            Action array for the robot.

        Raises:
            RuntimeError: If no action is available and no fallback exists.
        """
        self._execution.maybe_request(observation)
        action = self._action_queue.pop_or_none()
        if action is None:
            if self._last_action is not None:
                return self._last_action
            msg = "No action available from policy and no previous action to hold"
            raise RuntimeError(msg)
        self._last_action = action
        return action

    def stop(self) -> None:
        """Stop execution."""
        self._execution.stop()

    def reset(self) -> None:
        """Reset for a new episode."""
        self._action_queue.clear()
        self._model.reset()
        self._last_action = None
