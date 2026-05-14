# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Action queue for chunked policy output."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ActionQueue:
    """Stores action chunks and pops one action per tick.

    For chunked policies, the model produces N actions at once.
    The queue holds them and releases one per runtime tick.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the action queue.

        Args:
            max_size: Maximum number of actions to buffer.
        """
        self._queue: deque[np.ndarray] = deque(maxlen=max_size)

    def push_chunk(self, chunk: np.ndarray) -> None:
        """Push an action chunk (T, D) into the queue.

        Args:
            chunk: Array of shape ``(T, D)`` where T is chunk length.
        """
        for i in range(chunk.shape[0]):
            self._queue.append(chunk[i])

    def pop_or_none(self) -> np.ndarray | None:
        """Pop one action, or return None if empty."""
        if self._queue:
            return self._queue.popleft()
        return None

    def clear(self) -> None:
        """Clear all buffered actions."""
        self._queue.clear()

    @property
    def empty(self) -> bool:
        """Whether the queue has no actions."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
