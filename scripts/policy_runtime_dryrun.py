#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic Phase 3 runtime dry-run: any exported policy → PolicyRuntime.

Reads observations from a real robot + camera, runs inference through an
exported model, logs predicted actions — does NOT actuate (an interceptor
echoes the current state back as the action).

Works with any policy that exports through ``physicalai.inference.InferenceModel``
(MolmoAct2, Pi0.5, ACT, etc.). Supports both sync and async execution to
validate Phase 3 (FallbackAction + AsyncInferenceExecution + warmup).

Usage:
    python scripts/policy_runtime_dryrun.py \
        --export-dir /tmp/molmoact2_bf16 \
        --robot so101 --robot-port /dev/ttyACM0 \
        --robot-calibration /path/to/calibration.json \
        --camera uvc --camera-index 0 \
        --fps 2 --duration 30 \
        --execution async --warmup 2 \
        --task "pick up the yellow cube"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger


def build_robot(args: argparse.Namespace) -> Any:
    if args.robot == "so101":
        from physicalai.robot.so101 import SO101

        if args.resilient:
            return _ResilientSO101(
                port=args.robot_port,
                calibration=args.robot_calibration,
                role="follower",
            )
        return SO101(
            port=args.robot_port,
            calibration=args.robot_calibration,
            role="follower",
        )
    msg = f"Unknown robot type: {args.robot}"
    raise ValueError(msg)


def build_camera(args: argparse.Namespace) -> Any:
    from physicalai.capture.factory import create_camera

    if args.camera == "uvc":
        return create_camera(
            "uvc",
            device=args.camera_index,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            backend="v4l2",
        )
    msg = f"Unknown camera type: {args.camera}"
    raise ValueError(msg)


def build_execution(args: argparse.Namespace) -> Any:
    from physicalai.runtime import AsyncInferenceExecution, SyncInferenceExecution

    if args.execution == "sync":
        return SyncInferenceExecution(mode="chunk")
    if args.execution == "async":
        return AsyncInferenceExecution(refill_threshold=args.refill_threshold)
    msg = f"Unknown execution type: {args.execution}"
    raise ValueError(msg)


class _ResilientSO101:
    """SO101 wrapper that tolerates servo communication errors."""

    def __init__(self, port: str, calibration: str, role: str) -> None:
        from physicalai.robot.so101 import SO101

        self._robot = SO101(port=port, calibration=calibration, role=role)
        self._last_good_obs = None
        self._read_errors = 0

    def connect(self) -> None:
        self._robot.connect()

    def disconnect(self) -> None:
        self._robot.disconnect()

    def get_observation(self) -> Any:
        try:
            obs = self._robot.get_observation()
            self._last_good_obs = obs
            self._read_errors = 0
            return obs
        except ConnectionError:
            self._read_errors += 1
            if self._read_errors > 20 or self._last_good_obs is None:
                raise
            return self._last_good_obs

    def send_action(self, action: np.ndarray) -> None:
        try:
            self._robot.send_action(action)
        except ConnectionError:
            pass

    def is_connected(self) -> bool:
        return self._robot.is_connected()

    @property
    def joint_names(self) -> list[str]:
        return self._robot.joint_names


class _DryRunInterceptor:
    """Echoes observation state as the action so the robot never actuates."""

    def __init__(self) -> None:
        self.tick_count = 0
        self.tick_wall_ms: list[float] = []
        self._last_tick = time.perf_counter()

    def on_start(self) -> None:
        self._last_tick = time.perf_counter()

    def on_observation(self, observation: dict) -> None:
        pass

    def before_send_action(self, action: np.ndarray, observation: dict) -> np.ndarray:
        now = time.perf_counter()
        self.tick_wall_ms.append((now - self._last_tick) * 1000.0)
        self._last_tick = now
        self.tick_count += 1

        state = observation.get("state")
        if self.tick_count % 5 == 0 or self.tick_count <= 3:
            logger.info(
                f"tick={self.tick_count:3d} "
                f"action={np.round(action, 3).tolist()} "
                f"state={np.round(state, 3).tolist() if state is not None else 'N/A'}",
            )
        return state.copy() if state is not None else action

    def on_action_sent(self, action: np.ndarray, observation: dict) -> None:
        pass

    def on_error(self, error: BaseException, observation: dict) -> None:
        logger.warning(f"Runtime error at tick={self.tick_count}: {error!r}")

    def on_stop(self) -> None:
        pass


def _maybe_cast_bfloat16(model: Any) -> None:
    try:
        import torch

        adapter = getattr(model, "adapter", None)
        if adapter is None:
            return
        policy = getattr(adapter, "_policy", None)
        if policy is not None and hasattr(policy, "to"):
            adapter._policy = policy.to(dtype=torch.bfloat16)
            logger.info("Cast policy to bfloat16")
    except ImportError:
        pass


def run(args: argparse.Namespace) -> None:
    from physicalai.inference import InferenceModel
    from physicalai.runtime import HoldStateFallback, PolicyRuntime

    logger.info(f"[1/4] Loading model from {args.export_dir}")
    model = InferenceModel.load(args.export_dir, device=args.device)
    if args.bfloat16:
        _maybe_cast_bfloat16(model)
    logger.info(f"       backend={model.backend} device={model.device}")

    logger.info(f"[2/4] Building robot={args.robot} camera={args.camera}")
    robot = build_robot(args)
    robot.connect()
    camera = build_camera(args)
    camera.connect()

    logger.info(
        f"[3/4] Building runtime: execution={args.execution} fps={args.fps} "
        f"warmup={args.warmup}",
    )
    execution = build_execution(args)
    interceptor = _DryRunInterceptor()
    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=execution,
        fps=args.fps,
        cameras={"top": camera},
        callbacks=[interceptor],
        fallback=HoldStateFallback() if args.execution == "async" else None,
    )

    logger.info(f"[4/4] Running for {args.duration}s")
    try:
        model.reset()
        if args.warmup > 0:
            sample = None
            if args.task is not None:
                sample = _build_sample_with_task(robot, camera, args.task)
            runtime.warmup(sample, n=args.warmup)
        runtime.run(duration_s=args.duration)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
    finally:
        camera.disconnect()
        robot.disconnect()
        _print_summary(interceptor, execution)


def _build_sample_with_task(robot: Any, camera: Any, task: str) -> dict:
    obs = robot.get_observation()
    sample: dict = {
        "state": obs.joint_positions,
        "timestamp": obs.timestamp,
        "frame_index": 0,
        "task": task,
    }
    if obs.images is not None:
        sample["images"] = obs.images
    sample.setdefault("images", {})["top"] = camera.read_latest().data
    return sample


def _print_summary(interceptor: _DryRunInterceptor, execution: Any) -> None:
    logger.info("=" * 60)
    logger.info(f"Total ticks: {interceptor.tick_count}")
    if interceptor.tick_wall_ms:
        arr = np.asarray(interceptor.tick_wall_ms)
        logger.info(
            f"Tick wall-clock ms: "
            f"mean={arr.mean():.1f} p50={np.percentile(arr, 50):.1f} "
            f"p99={np.percentile(arr, 99):.1f} max={arr.max():.1f}",
        )
    if hasattr(execution, "inference_count"):
        logger.info(f"Inferences: {execution.inference_count}")
        if execution.inference_latency_ms:
            lat = np.asarray(execution.inference_latency_ms)
            logger.info(
                f"Inference latency ms: mean={lat.mean():.1f} "
                f"p99={np.percentile(lat, 99):.1f}",
            )
        logger.info(f"Transient failures: {execution.transient_failure_count}")
    logger.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bfloat16", action="store_true", help="Cast policy to bfloat16 (CUDA VRAM)")

    parser.add_argument("--robot", choices=["so101"], default="so101")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-calibration", required=False)
    parser.add_argument("--resilient", action="store_true", help="Tolerate servo errors (damaged joints)")

    parser.add_argument("--camera", choices=["uvc"], default="uvc")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)

    parser.add_argument("--execution", choices=["sync", "async"], default="async")
    parser.add_argument("--refill-threshold", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup inferences (0=disable)")
    parser.add_argument("--task", default=None, help="Task instruction for VLA policies")

    args = parser.parse_args()
    if not args.export_dir.exists():
        logger.error(f"Export dir {args.export_dir} not found")
        sys.exit(1)
    run(args)


if __name__ == "__main__":
    main()
