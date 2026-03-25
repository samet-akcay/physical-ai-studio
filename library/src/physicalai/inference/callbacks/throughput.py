# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Throughput monitor callback for inference performance tracking.

Tracks predictions per second over a sliding window.  Useful for
monitoring sustained throughput in deployment and detecting degradation.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, override

from physicalai.inference.callbacks.base import Callback

_DEFAULT_WINDOW_SIZE = 100
_MIN_SAMPLES_FOR_THROUGHPUT = 2


class ThroughputMonitor(Callback):
    """Track inference throughput (predictions/sec) over a sliding window.

    Attributes:
        throughput: Current throughput in predictions per second,
            computed over the most recent *window_size* predictions.
        total_predictions: Total number of predictions recorded.

    Examples:
        >>> monitor = ThroughputMonitor(window_size=50)
        >>> model = InferenceModel.load("./exports/act", callbacks=[monitor])
        >>> for obs in observations:
        ...     model.select_action(obs)
        >>> print(f"{monitor.throughput:.1f} predictions/sec")
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW_SIZE) -> None:
        """Initialise the throughput monitor.

        Args:
            window_size: Number of recent predictions to consider
                when computing throughput.  Defaults to 100.
        """
        self._window_size = window_size
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self.total_predictions: int = 0
        self.throughput: float = 0.0

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Record prediction timestamp and recompute throughput."""
        self._timestamps.append(time.perf_counter())
        self.total_predictions += 1

        if len(self._timestamps) >= _MIN_SAMPLES_FOR_THROUGHPUT:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self.throughput = (len(self._timestamps) - 1) / elapsed

    @override
    def __repr__(self) -> str:
        return (
            f"ThroughputMonitor("
            f"throughput={self.throughput:.1f}/s, "
            f"total={self.total_predictions}, "
            f"window={self._window_size})"
        )
