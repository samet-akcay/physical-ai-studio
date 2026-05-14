# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""PolicyRuntime convenience factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from physicalai.runtime.action_queue import ActionQueue
from physicalai.runtime.policy_controller import PolicyController
from physicalai.runtime.runtime import RobotRuntime

if TYPE_CHECKING:
    from physicalai.capture.camera import Camera
    from physicalai.inference.model import InferenceModel
    from physicalai.robot.interface import Robot

    from .callbacks import RuntimeCallback
    from .execution import InferenceExecution
    from .fallback import FallbackAction
    from .safety import SafetyLayer


def PolicyRuntime(  # noqa: N802
    *,
    robot: Robot,
    model: InferenceModel,
    execution: InferenceExecution,
    fps: float,
    action_queue: ActionQueue | None = None,
    cameras: dict[str, Camera] | None = None,
    callbacks: list[RuntimeCallback] | None = None,
    safety: SafetyLayer | None = None,
    fallback: FallbackAction | None = None,
    return_to_home: bool = False,
) -> RobotRuntime:
    """Create a RobotRuntime with a PolicyController.

    One-line factory for policy-only deployment.

    Args:
        robot: Robot instance.
        model: Loaded InferenceModel.
        execution: Inference execution strategy.
        fps: Loop frequency in Hz.
        action_queue: Optional action queue (created if None).
        cameras: Optional external cameras.
        callbacks: Optional runtime callbacks.
        safety: Optional safety layer.
        fallback: Optional fallback used by PolicyController when the queue is
            empty and no previous action exists. Required for async execution
            without warmup.
        return_to_home: Whether to return to home on shutdown.

    Returns:
        Configured RobotRuntime instance.
    """
    controller = PolicyController(
        model=model,
        execution=execution,
        action_queue=action_queue,
        fallback=fallback,
    )
    return RobotRuntime(
        robot=robot,
        controller=controller,
        fps=fps,
        cameras=cameras,
        callbacks=callbacks,
        safety=safety,
        return_to_home=return_to_home,
    )
