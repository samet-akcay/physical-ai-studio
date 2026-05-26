#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end dry-run: MolmoAct2 export(torch) → InferenceModel → PolicyRuntime.

Reads real observations (camera + joint state) from SO-101, runs inference
through the exported model, logs predicted actions, and intercepts
send_action to keep the robot stationary (state-as-action).

Usage::

    # Export once (reuse the directory)
    python library/scripts/molmoact2_dry_run.py --export-only

    # Run runtime with existing export
    python library/scripts/molmoact2_dry_run.py --export-dir /tmp/molmoact2_bf16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
CALIBRATION = (
    "/home/sakcay/.cache/physicalai/robots/8d8353a0-fc8c-49aa-b5f1-33290f726698/"
    "calibrations/6e6303f3-495a-4e75-8d30-edcc8932c7dd.json"
)
PORT = "/dev/ttyACM0"
CAMERA_INDEX = 0
FPS = 2.0
DURATION_S = 10.0
EXPORT_DIR = Path("/tmp/molmoact2_bf16")
TASK_PROMPT = "pick up the red block"


def export_model(export_dir: Path) -> None:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    from physicalai.policies.lerobot import MolmoAct2

    print(f"Loading MolmoAct2 from {REPO_ID}...")
    config = MolmoAct2Config(
        checkpoint_path=REPO_ID,
        norm_tag=NORM_TAG,
        inference_action_mode="continuous",
        enable_inference_cuda_graph=False,
        model_dtype="bfloat16",
        input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,))},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    policy = MolmoAct2(config=config)
    policy = policy.to(dtype=torch.bfloat16)

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting (bfloat16) -> {export_dir}")
    policy.export(export_dir, backend="torch")
    print(f"Done. Files: {sorted(p.name for p in export_dir.iterdir())}")


def _inject_task(observation: dict[str, Any]) -> dict[str, Any]:
    observation = dict(observation)
    observation.setdefault("task", [TASK_PROMPT])
    return observation


def run_runtime(export_dir: Path) -> None:
    print(f"[1/2] Loading InferenceModel.load({export_dir})...")
    from physicalai.inference import InferenceModel

    model = InferenceModel.load(export_dir, device="cuda")

    if model.adapter._policy is not None:
        model.adapter._policy = model.adapter._policy.to(dtype=torch.bfloat16)

    original_predict_action_chunk = model.predict_action_chunk

    def predict_with_task(observation: dict[str, Any]) -> np.ndarray:
        return original_predict_action_chunk(_inject_task(observation))

    model.predict_action_chunk = predict_with_task  # type: ignore[method-assign]

    print(f"       backend={model.backend} device={model.device}")

    print(f"[2/2] PolicyRuntime dry-run ({DURATION_S}s @ {FPS} Hz, task={TASK_PROMPT!r})...")
    from physicalai.capture.factory import create_camera
    from physicalai.robot.so101 import SO101
    from physicalai.runtime import PolicyRuntime, RuntimeCallback, SyncExecution

    class DryRunLogger(RuntimeCallback):
        def __init__(self) -> None:
            self.ticks = 0

        def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray | None:
            self.ticks += 1
            print(f"  step={step:4d} action={np.round(action, 4).tolist()}")
            return None

        def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
            return None

        def on_hold(self, *, step: int, holds: int) -> None:
            if holds == 1:
                print(f"  step={step:4d} [hold]")

    class DryRunNoActuate(RuntimeCallback):
        """Replace policy action with the last observed state so the arm stays put."""

        def __init__(self, robot: SO101) -> None:
            self._robot = robot

        def before_send_action(self, *, action: np.ndarray, step: int) -> np.ndarray | None:
            obs = self._robot.get_observation()
            state = np.asarray(obs.state, dtype=action.dtype)
            return state

        def on_action_sent(self, *, action: np.ndarray, step: int) -> None:
            return None

        def on_hold(self, *, step: int, holds: int) -> None:
            return None

    robot = SO101(port=PORT, calibration=CALIBRATION, role="follower")
    camera = create_camera("uvc", device=CAMERA_INDEX, width=640, height=480, fps=30, backend="v4l2")

    logger = DryRunLogger()
    no_actuate = DryRunNoActuate(robot)

    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=SyncExecution(),
        fps=FPS,
        cameras={"top": camera},
        callbacks=[no_actuate, logger],
    )

    try:
        with runtime:
            model.reset()
            stats = runtime.run(duration_s=DURATION_S)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return

    print(f"\nDry-run complete. ticks={logger.ticks} stats={stats}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true", help="Export model and exit")
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR, help="Path to exported model")
    args = parser.parse_args()

    if args.export_only:
        export_model(args.export_dir)
        return

    if not args.export_dir.exists():
        print(f"Export dir {args.export_dir} not found. Run with --export-only first.")
        sys.exit(1)
    run_runtime(args.export_dir)


if __name__ == "__main__":
    main()
