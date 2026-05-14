# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference execution strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np

    from physicalai.inference.model import InferenceModel
    from physicalai.runtime.action_queue import ActionQueue


class InferenceExecution(Protocol):
    """Controls when and where inference runs."""

    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None:
        """Bind to the action queue and model."""
        ...

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Potentially trigger inference based on observation."""
        ...

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Warm up the model with sample observations."""
        ...

    def stop(self) -> None:
        """Stop execution and release resources."""
        ...


class SyncInferenceExecution:
    """Synchronous inference execution — runs in the runtime thread.

    Args:
        mode: ``"single_action"`` calls ``select_action()`` each tick.
            ``"chunk"`` calls ``predict_action_chunk()`` when the queue is empty.
    """

    def __init__(self, mode: str = "chunk") -> None:
        if mode not in ("single_action", "chunk"):
            msg = f"mode must be 'single_action' or 'chunk', got '{mode}'"
            raise ValueError(msg)
        self._mode = mode
        self._action_queue: ActionQueue | None = None
        self._model: InferenceModel | None = None

    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None:
        """Bind to action queue and model."""
        self._action_queue = action_queue
        self._model = model

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Run inference synchronously when the queue needs refilling."""
        if self._model is None or self._action_queue is None:
            return

        if self._mode == "single_action":
            action: np.ndarray = self._model.select_action(observation)
            
            import numpy as np

            self._action_queue.push_chunk(np.expand_dims(action, axis=0))
        elif self._mode == "chunk" and self._action_queue.empty:
            outputs = self._model(observation)
            from physicalai.inference.constants import ACTION

            chunk = outputs[ACTION]
            if chunk.ndim == 1:
                import numpy as np

                chunk = np.expand_dims(chunk, axis=0)
            elif chunk.ndim == 3:  # noqa: PLR2004
                chunk = chunk[0]
            self._action_queue.push_chunk(chunk)

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Run warmup inferences (discarded)."""
        if self._model is None:
            return
        for _ in range(n):
            self._model(sample_observation)

    def stop(self) -> None:
        """No-op for sync execution."""
