# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference execution strategies."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel
    from physicalai.runtime.action_queue import ActionQueue


_TRANSIENT_EXC: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, _TRANSIENT_EXC):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg


def _normalize_chunk(outputs: Any) -> np.ndarray:
    from physicalai.inference.constants import ACTION

    chunk = outputs[ACTION] if isinstance(outputs, dict) else outputs
    if chunk.ndim == 1:
        chunk = np.expand_dims(chunk, axis=0)
    elif chunk.ndim == 3:
        chunk = chunk[0]
    return chunk


class InferenceExecution(Protocol):
    """Controls when and where inference runs."""

    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None:
        """Bind to the action queue and model and begin execution if applicable."""
        ...

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Potentially trigger inference based on observation. Must not block."""
        ...

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Run blocking warm-up inferences and push their chunks to the queue."""
        ...

    def stop(self) -> None:
        """Stop execution and release resources."""
        ...


class SyncInferenceExecution:
    """Synchronous inference execution — runs in the runtime thread.

    Args:
        mode: ``"single_action"`` calls ``select_action()`` each tick.
            ``"chunk"`` calls the model when the queue is empty.
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
            action = self._model.select_action(observation)
            self._action_queue.push_chunk(np.expand_dims(action, axis=0))
        elif self._mode == "chunk" and self._action_queue.empty:
            chunk = _normalize_chunk(self._model(observation))
            self._action_queue.push_chunk(chunk)

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Run warmup inferences and push chunks to the queue."""
        if self._model is None or self._action_queue is None:
            return
        for _ in range(n):
            if self._mode == "single_action":
                action = self._model.select_action(sample_observation)
                self._action_queue.push_chunk(np.expand_dims(action, axis=0))
            else:
                chunk = _normalize_chunk(self._model(sample_observation))
                self._action_queue.push_chunk(chunk)

    def stop(self) -> None:
        """No-op for sync execution."""


class AsyncInferenceExecution:
    """Asynchronous inference on a worker thread.

    The runtime tick never blocks on the model. Observations are passed via a
    latest-wins single-slot mailbox; stale observations are dropped. The
    worker triggers inference when the action queue depth falls at or below
    ``refill_threshold``.

    Args:
        refill_threshold: Trigger inference when ``len(queue) <= refill_threshold``.
        max_inflight: Maximum concurrent inferences (1 for single-GPU systems).
        backoff_schedule_s: Sleep durations between transient-failure retries.
        max_consecutive_failures: After this many consecutive failures, escalate to fatal.
        shutdown_timeout_s: Seconds to wait for in-flight inference on stop().

    Metrics:
        ``inference_count``: completed inferences.
        ``inference_latency_ms``: rolling window (last 100) of inference latencies.
        ``transient_failure_count``: transient failures observed.
        ``fatal_exception``: stored fatal exception (re-raised on next maybe_request).
    """

    def __init__(
        self,
        refill_threshold: int = 4,
        max_inflight: int = 1,
        backoff_schedule_s: tuple[float, ...] = (0.1, 1.0),
        max_consecutive_failures: int = 3,
        shutdown_timeout_s: float = 10.0,
    ) -> None:
        if refill_threshold < 0:
            msg = f"refill_threshold must be >= 0, got {refill_threshold}"
            raise ValueError(msg)
        if max_inflight < 1:
            msg = f"max_inflight must be >= 1, got {max_inflight}"
            raise ValueError(msg)

        self._refill_threshold = refill_threshold
        self._max_inflight = max_inflight
        self._backoff_schedule_s = backoff_schedule_s
        self._max_consecutive_failures = max_consecutive_failures
        self._shutdown_timeout_s = shutdown_timeout_s

        self._action_queue: ActionQueue | None = None
        self._model: InferenceModel | None = None

        self._latest_obs: dict[str, Any] | None = None
        self._obs_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._inflight = 0
        self._inflight_lock = threading.Lock()
        self._consecutive_failures = 0
        self._fatal_exception: BaseException | None = None

        self.inference_count = 0
        self.inference_latency_ms: deque[float] = deque(maxlen=100)
        self.transient_failure_count = 0

    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None:
        """Bind queue and model, spawn the worker thread."""
        self._action_queue = action_queue
        self._model = model
        self._stop.clear()
        self._fatal_exception = None
        self._consecutive_failures = 0

        self._thread = threading.Thread(
            target=self._worker_loop,
            name="inference-worker",
            daemon=True,
        )
        self._thread.start()

    def maybe_request(self, observation: dict[str, Any]) -> None:
        """Drop the latest observation in the worker's mailbox. Non-blocking.

        Re-raises any fatal worker exception on the runtime thread.
        """
        if self._fatal_exception is not None:
            exc, self._fatal_exception = self._fatal_exception, None
            raise exc

        with self._obs_lock:
            self._latest_obs = observation
        self._wake.set()

    def warmup(self, sample_observation: dict[str, Any], n: int = 2) -> None:
        """Run blocking inferences on the calling thread to pre-fill the queue.

        Should be invoked BEFORE start() or after start() if the worker is idle.
        Bypasses the worker thread to ensure the queue has actions before the
        runtime begins ticking.
        """
        if self._model is None or self._action_queue is None:
            msg = "warmup() requires start() to have been called first"
            raise RuntimeError(msg)
        for _ in range(n):
            chunk = _normalize_chunk(self._model(sample_observation))
            self._action_queue.push_chunk(chunk)

    def stop(self) -> None:
        """Signal worker to stop; wait for in-flight inference up to timeout."""
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=self._shutdown_timeout_s)
            if self._thread.is_alive():
                logger.warning(
                    f"Inference worker did not stop within {self._shutdown_timeout_s}s",
                )
            self._thread = None

    def _take_latest_obs(self) -> dict[str, Any] | None:
        with self._obs_lock:
            obs, self._latest_obs = self._latest_obs, None
            return obs

    def _should_refill(self) -> bool:
        if self._action_queue is None:
            return False
        if len(self._action_queue) > self._refill_threshold:
            return False
        with self._inflight_lock:
            return self._inflight < self._max_inflight

    def _worker_loop(self) -> None:
        pending_retry: dict[str, Any] | None = None
        while not self._stop.is_set():
            if pending_retry is None:
                self._wake.wait(timeout=0.1)
                self._wake.clear()
            if self._stop.is_set():
                break
            if not self._should_refill():
                pending_retry = None
                continue
            obs = pending_retry if pending_retry is not None else self._take_latest_obs()
            if obs is None:
                continue
            failed_before = self._consecutive_failures
            self._run_one_inference(obs)
            if (
                self._consecutive_failures > failed_before
                and self._fatal_exception is None
                and not self._stop.is_set()
            ):
                pending_retry = obs
            else:
                pending_retry = None

    def _run_one_inference(self, observation: dict[str, Any]) -> None:
        with self._inflight_lock:
            self._inflight += 1
        try:
            t0 = time.perf_counter()
            chunk = _normalize_chunk(self._model(observation))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            assert self._action_queue is not None
            self._action_queue.push_chunk(chunk)

            self.inference_count += 1
            self.inference_latency_ms.append(latency_ms)
            self._consecutive_failures = 0

        except BaseException as exc:  # noqa: BLE001
            self._handle_worker_exception(exc)
        finally:
            with self._inflight_lock:
                self._inflight -= 1

    def _handle_worker_exception(self, exc: BaseException) -> None:
        self._consecutive_failures += 1
        if not _is_transient(exc):
            logger.exception("Inference worker fatal exception")
            self._fatal_exception = exc
            self._stop.set()
            return

        self.transient_failure_count += 1
        logger.warning(
            f"Transient inference failure ({self._consecutive_failures}/"
            f"{self._max_consecutive_failures}): {exc!r}",
        )

        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.error("Transient failures exceeded threshold — escalating to fatal")
            self._fatal_exception = exc
            self._stop.set()
            return

        idx = min(self._consecutive_failures - 1, len(self._backoff_schedule_s) - 1)
        backoff = self._backoff_schedule_s[idx]
        time.sleep(backoff)
