# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyController — adapts InferenceModel into a Controller."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.runtime.action_queue import ActionQueue

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.model import InferenceModel
    from physicalai.runtime.execution import InferenceExecution
    from physicalai.runtime.fallback import FallbackAction


class PolicyController:
    """Adapts InferenceModel + InferenceExecution + ActionQueue into a Controller.

    Args:
        model: Loaded inference model.
        execution: Inference execution strategy (sync/async/remote).
        action_queue: Optional pre-configured action queue.
        fallback: Optional fallback used when the queue is empty and no
            previous action exists. Required for async execution to avoid
            bootstrap errors on the first tick.
    """

    def __init__(
        self,
        model: InferenceModel,
        execution: InferenceExecution,
        action_queue: ActionQueue | None = None,
        fallback: FallbackAction | None = None,
    ) -> None:
        self._model = model
        self._execution = execution
        self._action_queue = action_queue if action_queue is not None else ActionQueue()
        self._fallback = fallback
        self._last_action: np.ndarray | None = None
        self._fallback_count = 0

    @property
    def fallback_count(self) -> int:
        """Number of times the fallback has been invoked since start."""
        return self._fallback_count

    def start(self) -> None:
        """Start the execution strategy."""
        self._execution.start(self._action_queue, self._model)

    def update(self, observation: dict[str, Any]) -> np.ndarray:
        """Request inference if needed and pop one action.

        Returns:
            Action array. Resolution order: queue → last_action → fallback → raise.

        Raises:
            RuntimeError: If no action is available and no fallback is configured.
        """
        self._execution.maybe_request(observation)
        action = self._action_queue.pop_or_none()
        if action is not None:
            self._last_action = action
            return action

        if self._last_action is not None:
            return self._last_action

        if self._fallback is not None:
            self._fallback_count += 1
            logger.debug(f"PolicyController: using fallback (count={self._fallback_count})")
            fallback_action = self._fallback.action(observation)
            self._last_action = fallback_action
            return fallback_action

        msg = (
            "PolicyController has no action: queue empty, no previous action, "
            "and no fallback configured. Pass fallback=HoldStateFallback() "
            "for async execution."
        )
        raise RuntimeError(msg)

    def stop(self) -> None:
        """Stop execution."""
        self._execution.stop()

    def reset(self) -> None:
        """Reset for a new episode."""
        self._action_queue.clear()
        self._model.reset()
        self._last_action = None
        self._fallback_count = 0
