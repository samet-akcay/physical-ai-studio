# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Timing callback for inference performance profiling.

Records wall-clock latency of each ``select_action`` / ``__call__``
invocation.  Useful for benchmarking and detecting latency regressions.
"""

from __future__ import annotations

import time
from typing import Any, override

from physicalai.inference.callbacks.base import Callback


class TimingCallback(Callback):
    """Record prediction latency in milliseconds.

    Attributes:
        last_duration_ms: Duration of the most recent prediction (ms).
        history: List of all recorded durations (ms).

    Examples:
        >>> timer = TimingCallback()
        >>> model = InferenceModel.load("./exports/act", callbacks=[timer])
        >>> action = model.select_action(obs)
        >>> print(f"Took {timer.last_duration_ms:.1f} ms")
    """

    def __init__(self) -> None:
        """Initialise timing state."""
        self.last_duration_ms: float = 0.0
        self.history: list[float] = []
        self._start_time: float = 0.0

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        """Record the prediction start time."""
        self._start_time = time.perf_counter()

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Compute and store the prediction duration."""
        ms_per_second = 1000.0
        self.last_duration_ms = (time.perf_counter() - self._start_time) * ms_per_second
        self.history.append(self.last_duration_ms)

    @override
    def __repr__(self) -> str:
        """Return string representation with latest timing."""
        return f"TimingCallback(last={self.last_duration_ms:.1f}ms, calls={len(self.history)})"
