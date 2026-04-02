# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action-chunking inference runner (decorator).

Wraps any ``InferenceRunner`` to add temporal action buffering.  The inner
runner produces an output dict whose action value has shape
``(batch, horizon, action_dim)``.  This wrapper queues the individual
timesteps, dispensing one per call.

This is the GoF Decorator pattern: ``ActionChunking`` *is* an
``InferenceRunner`` and *has* an ``InferenceRunner``.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002

from physicalai.inference.constants import ACTION
from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter


class ActionChunking(InferenceRunner):
    """Wrap a runner with temporal action buffering.

    On the first call (or when the queue is empty), delegates to the
    inner runner which returns an output dict containing an action with
    shape ``(batch, chunk_size, action_dim)``.  All chunk steps are
    enqueued and one is returned.  Subsequent calls pop from the queue
    without running inference.

    Args:
        runner: The inner runner to delegate inference to. If ``None``,
            a :class:`SinglePass` is created lazily on first use.
        chunk_size: Number of actions per chunk.  Must match the inner
            runner's output temporal dimension.
        n_action_steps: Number of actions to actually enqueue from each
            chunk.  Defaults to ``chunk_size``.
        action_key: Key in the runner output dict that holds the action.

    Examples:
        Wrap a single-pass runner with action chunking:

        >>> runner = ActionChunking(chunk_size=10)
        >>> outputs = runner.run(adapters, inputs)  # runs inference, queues 10 actions
        >>> outputs = runner.run(adapters, inputs)  # pops from queue, no inference

        Compose with any runner (e.g. future flow-matching):

        >>> runner = ActionChunking(FlowMatching(num_steps=20), chunk_size=5)
    """

    def __init__(
        self,
        runner: InferenceRunner | None = None,
        chunk_size: int = 1,
        n_action_steps: int | None = None,
        action_key: str = ACTION,
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Initialize with an inner runner and chunk configuration.

        Args:
            runner: The inner runner to wrap (or ``None`` for lazy SinglePass).
            chunk_size: Number of actions per chunk.
            n_action_steps: Number of chunk steps to queue per refill.
            action_key: Key for the action tensor in the output dict.
            **kwargs: Extra manifest parameters accepted for compatibility.
        """
        self.runner = runner
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps or chunk_size
        self.action_key = action_key
        self._action_queue: deque[np.ndarray] = deque()

    def run(
        self,
        adapters: dict[str, RuntimeAdapter],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the next action, running inference only when queue is empty.

        Args:
            adapters: Named runtime adapters, passed to the wrapped runner.
            inputs: Pre-processed model inputs.

        Returns:
            Output dict with a single action of shape ``(batch_size, action_dim)``.
        """
        if len(self._action_queue) > 0:
            return {self.action_key: self._action_queue.popleft()}

        if self.runner is None:
            from physicalai.inference.runners.single_pass import SinglePass  # noqa: PLC0415

            self.runner = SinglePass()

        outputs = self.runner.run(adapters, inputs)
        chunk = outputs[self.action_key]

        # Squeeze leading batch dim if present: (1, T, D) -> (T, D)
        if chunk.ndim == 3 and chunk.shape[0] == 1:  # noqa: PLR2004
            chunk = chunk[0]

        n_steps = min(self.n_action_steps, len(chunk))
        for i in range(n_steps):
            self._action_queue.append(chunk[i])

        return {self.action_key: self._action_queue.popleft()}

    def reset(self) -> None:
        """Clear the action queue and reset the inner runner."""
        self._action_queue.clear()
        if self.runner is not None:
            self.runner.reset()

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return f"{self.__class__.__name__}(runner={self.runner!r}, chunk_size={self.chunk_size})"
