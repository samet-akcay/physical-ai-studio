# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RobotRuntime — the synchronous robot control loop."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from physicalai.runtime.safety import SafetyViolationError

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera
    from physicalai.robot.interface import Robot

    from .callbacks import RuntimeCallback
    from .controller import Controller
    from .safety import SafetyLayer


class RobotRuntime:
    """Single-rate synchronous robot control loop.

    Drives one robot at a fixed FPS using one controller.

    Args:
        robot: Robot instance satisfying the Robot protocol.
        controller: Action-selection controller.
        fps: Loop frequency in Hz.
        cameras: Optional external cameras keyed by name.
        callbacks: Sequence of runtime callbacks.
        safety: Optional safety layer for action filtering.
        return_to_home: Whether to return robot to home on shutdown.
    """

    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        fps: float,
        cameras: dict[str, Camera] | None = None,
        callbacks: list[RuntimeCallback] | None = None,
        safety: SafetyLayer | None = None,
        *,
        return_to_home: bool = False,
    ) -> None:
        self._robot = robot
        self._controller = controller
        self._fps = fps
        self._tick_duration = 1.0 / fps
        self._cameras = cameras or {}
        self._callbacks = callbacks or []
        self._safety = safety
        self._return_to_home = return_to_home
        self._running = False
        self._controller_started = False
        self._stop_flag = threading.Event()
        self._controller_lock = threading.Lock()

    @property
    def fps(self) -> float:
        """Target loop frequency."""
        return self._fps

    @property
    def running(self) -> bool:
        """Whether the loop is currently running."""
        return self._running

    def run(self, *, duration_s: float | None = None) -> None:
        """Run the control loop.

        Blocks until ``stop()`` is called, duration expires, or a fatal error occurs.

        Args:
            duration_s: Maximum duration in seconds. None = run until stopped.
        """
        self._stop_flag.clear()
        self._running = True

        with self._controller_lock:
            if not self._controller_started:
                self._controller.start()
                self._controller_started = True

        for cb in self._callbacks:
            cb.on_start()

        frame_index = 0
        start_time = time.monotonic()
        last_safe_action = None

        try:
            while not self._stop_flag.is_set():
                if duration_s is not None and (time.monotonic() - start_time) >= duration_s:
                    break

                tick_start = time.monotonic()

                try:
                    obs = self._robot.get_observation()
                except Exception:
                    logger.exception("Robot observation failed — shutting down")
                    break

                observation: dict[str, Any] = {
                    "state": obs.joint_positions,
                    "timestamp": obs.timestamp,
                    "frame_index": frame_index,
                }
                if obs.images is not None:
                    observation["images"] = obs.images
                if obs.sensor_data is not None:
                    observation["sensor_data"] = obs.sensor_data

                for name, camera in self._cameras.items():
                    observation.setdefault("images", {})[name] = camera.read_latest().data

                for cb in self._callbacks:
                    cb.on_observation(observation)

                try:
                    with self._controller_lock:
                        action = self._controller.update(observation)
                except Exception as e:
                    logger.warning(f"Controller error: {e}")
                    for cb in self._callbacks:
                        cb.on_error(e, observation)
                    if last_safe_action is not None:
                        action = last_safe_action
                    else:
                        continue

                for cb in self._callbacks:
                    action = cb.before_send_action(action, observation)

                if self._safety is not None:
                    try:
                        action = self._safety.filter(action, observation)
                    except SafetyViolationError:
                        logger.error("Safety violation — shutting down")
                        break
                    except Exception as e:
                        logger.warning(f"Safety layer error (non-fatal): {e}")
                        for cb in self._callbacks:
                            cb.on_error(e, observation)
                        if last_safe_action is not None:
                            action = last_safe_action

                try:
                    self._robot.send_action(action)
                except Exception:
                    logger.exception("Robot send_action failed — shutting down")
                    break

                last_safe_action = action

                for cb in self._callbacks:
                    cb.on_action_sent(action, observation)

                frame_index += 1

                elapsed = time.monotonic() - tick_start
                sleep_time = self._tick_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            self._shutdown()

    def stop(self) -> None:
        """Signal the loop to stop. Thread-safe."""
        self._stop_flag.set()

    def warmup(
        self,
        sample_observation: dict[str, Any] | None = None,
        n: int = 2,
    ) -> None:
        """Pre-warm the controller by running ``n`` blocking inferences.

        If ``sample_observation`` is None, captures one from the robot. The
        controller must implement ``warmup()``; controllers without it are
        skipped silently.
        """
        if not hasattr(self._controller, "warmup"):
            logger.debug("Controller has no warmup() — skipping")
            return

        if sample_observation is None:
            obs = self._robot.get_observation()
            sample_observation = {
                "state": obs.joint_positions,
                "timestamp": obs.timestamp,
                "frame_index": 0,
            }
            if obs.images is not None:
                sample_observation["images"] = obs.images
            if obs.sensor_data is not None:
                sample_observation["sensor_data"] = obs.sensor_data
            for name, camera in self._cameras.items():
                sample_observation.setdefault("images", {})[name] = camera.read_latest().data

        logger.info(f"Warming up controller with {n} inference(s)")
        with self._controller_lock:
            if not self._controller_started:
                self._controller.start()
                self._controller_started = True
            self._controller.warmup(sample_observation, n=n)

    def swap_controller(self, controller: Controller) -> None:
        """Replace the active controller. Thread-safe.

        Args:
            controller: New controller to use.
        """
        with self._controller_lock:
            self._controller.stop()
            self._controller = controller
            self._controller.start()
            self._controller_started = True

    def _shutdown(self) -> None:
        """Perform safe shutdown sequence."""
        self._running = False
        with self._controller_lock:
            self._controller.stop()
            self._controller_started = False
        for cb in self._callbacks:
            cb.on_stop()
        if self._return_to_home and hasattr(self._robot, "go_to_home"):
            self._robot.go_to_home()
        self._robot.disconnect()
