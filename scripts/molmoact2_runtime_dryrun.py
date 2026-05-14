#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end dry-run: MolmoAct2 export(torch) → InferenceModel → PolicyRuntime.

Reads real observations (camera + joint state) from SO-101, runs inference
through the exported model, logs predicted actions — does NOT actuate.

Usage:
    # Step 1: Export (run once, reuse the export dir)
    python scripts/molmoact2_runtime_dryrun.py --export-only

    # Step 2: Run runtime with existing export
    python scripts/molmoact2_runtime_dryrun.py --export-dir /tmp/molmoact2_bf16
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"
CALIBRATION = "/home/sakcay/.cache/physicalai/robots/8d8353a0-fc8c-49aa-b5f1-33290f726698/calibrations/6e6303f3-495a-4e75-8d30-edcc8932c7dd.json"
PORT = "/dev/ttyACM0"
CAMERA_INDEX = 0
FPS = 2.0
DURATION_S = 10.0
EXPORT_DIR = Path("/tmp/molmoact2_bf16")


def export_model(export_dir: Path) -> None:
    from physicalai.policies.molmoact2 import MolmoAct2

    print(f"Loading MolmoAct2 from {REPO_ID}...")
    policy = MolmoAct2.from_checkpoint(REPO_ID, norm_tag=NORM_TAG)
    policy = policy.to(dtype=torch.bfloat16)

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting (bfloat16) → {export_dir}")
    policy.export(export_dir, backend="torch")
    print(f"Done. Files: {list(export_dir.iterdir())}")


def run_runtime(export_dir: Path) -> None:
    print(f"[1/2] Loading via InferenceModel.load({export_dir})...")
    from physicalai.inference import InferenceModel

    model = InferenceModel.load(export_dir, device="cuda")

    # Cast to bfloat16 to fit in 24GB VRAM
    if model.adapter._policy is not None:
        model.adapter._policy = model.adapter._policy.to(dtype=torch.bfloat16)

    print(f"       Backend: {model.backend}, Device: {model.device}")

    print(f"[2/2] Running PolicyRuntime dry-run ({DURATION_S}s at {FPS} Hz)...")
    from physicalai.robot.so101 import SO101
    from physicalai.runtime import PolicyRuntime, RuntimeCallback, SyncInferenceExecution

    class ResilientSO101:
        """Wraps SO101 to tolerate servo communication errors on damaged joints."""

        def __init__(self, port, calibration, role):
            self._robot = SO101(port=port, calibration=calibration, role=role)
            self._last_good_obs = None
            self._read_errors = 0

        def connect(self):
            self._robot.connect()

        def disconnect(self):
            self._robot.disconnect()

        def get_observation(self):
            try:
                obs = self._robot.get_observation()
                self._last_good_obs = obs
                self._read_errors = 0
                return obs
            except ConnectionError:
                self._read_errors += 1
                if self._read_errors > 20:
                    raise
                if self._last_good_obs is not None:
                    return self._last_good_obs
                raise

        def send_action(self, action):
            try:
                self._robot.send_action(action)
            except ConnectionError:
                pass

        def is_connected(self):
            return self._robot.is_connected()

        @property
        def joint_names(self):
            return self._robot.joint_names

    robot = ResilientSO101(port=PORT, calibration=CALIBRATION, role="follower")
    robot.connect()

    from physicalai.capture.factory import create_camera

    camera = create_camera("uvc", device=CAMERA_INDEX, width=640, height=480, fps=30, backend="v4l2")
    camera.connect()

    class DryRunLogger(RuntimeCallback):
        def __init__(self):
            self.tick_count = 0

        def on_observation(self, observation):
            pass

        def before_send_action(self, action, observation):
            self.tick_count += 1
            state = observation.get("state", None)
            print(
                f"  tick={self.tick_count:3d} "
                f"action={np.round(action, 4).tolist()} "
                f"state={np.round(state, 4).tolist() if state is not None else 'N/A'}"
            )
            return action

    class CameraInjector(RuntimeCallback):
        """Injects camera frame into observation before inference."""

        def on_observation(self, observation):
            frame = camera.read_latest()
            observation.setdefault("images", {})["top"] = frame.data

    class DryRunInterceptor(RuntimeCallback):
        """Prevents actual actuation by returning current state as action."""

        def before_send_action(self, action, observation):
            state = observation.get("state")
            if state is not None:
                return state.copy()
            return action

    logger = DryRunLogger()
    camera_cb = CameraInjector()
    interceptor = DryRunInterceptor()

    execution = SyncInferenceExecution(mode="chunk")

    runtime = PolicyRuntime(
        robot=robot,
        model=model,
        execution=execution,
        fps=FPS,
        callbacks=[camera_cb, logger, interceptor],
    )

    try:
        model.reset()
        runtime.run(duration_s=DURATION_S)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.disconnect()
        robot.disconnect()
        print(f"\nDry-run complete. {logger.tick_count} ticks executed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-only", action="store_true", help="Export model and exit")
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR, help="Path to exported model")
    args = parser.parse_args()

    if args.export_only:
        export_model(args.export_dir)
    else:
        if not args.export_dir.exists():
            print(f"Export dir {args.export_dir} not found. Run with --export-only first.")
            sys.exit(1)
        run_runtime(args.export_dir)


if __name__ == "__main__":
    main()
