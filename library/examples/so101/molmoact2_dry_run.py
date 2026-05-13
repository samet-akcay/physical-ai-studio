# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 SO-101 policy dry-run using PhysicalAI robot and camera APIs.

This is intentionally a small proof-of-concept, not the final runtime system.
It mirrors the planned ``RobotRuntime + PolicyController`` loop with explicit
steps: read robot, read cameras, run policy, optionally send the action.

Dry-run is the default. Use ``--actuate`` only after inspecting printed actions
and confirming your calibration, workspace, and emergency stop are ready.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from physicalai.capture.camera import Camera
from physicalai.capture.factory import create_camera
from physicalai.robot.so101 import SO101


REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"


def _physicalai_state_to_molmo(state_rad: np.ndarray) -> np.ndarray:
    """Convert PhysicalAI SO101 radians to LeRobot SO100/101 state convention."""
    state = np.asarray(state_rad, dtype=np.float32).copy()
    state[:5] = np.rad2deg(state[:5])
    return state


def _molmo_action_to_physicalai(action: np.ndarray) -> np.ndarray:
    """Convert MolmoAct2 SO100/101 absolute joint pose to PhysicalAI radians."""
    action = np.asarray(action, dtype=np.float32).copy()
    action[:5] = np.deg2rad(action[:5])
    return action


def _parse_camera_spec(spec: str) -> tuple[str, str, dict[str, object]]:
    """Parse ``name:type:key=value,...`` into ``create_camera`` arguments."""
    try:
        name, camera_type, raw_kwargs = spec.split(":", 2)
    except ValueError as e:
        msg = f"Invalid camera spec {spec!r}. Expected name:type:key=value,..."
        raise argparse.ArgumentTypeError(msg) from e

    kwargs: dict[str, object] = {}
    if raw_kwargs:
        for item in raw_kwargs.split(","):
            key, value = item.split("=", 1)
            kwargs[key] = int(value) if value.isdecimal() else value

    return name, camera_type, kwargs


def _connect_cameras(specs: Sequence[str]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}
    for spec in specs:
        name, camera_type, kwargs = _parse_camera_spec(spec)
        camera = create_camera(camera_type, **kwargs)
        camera.connect()
        cameras[name] = camera
    return cameras


def _read_images(cameras: dict[str, Camera]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for camera in cameras.values():
        frame = camera.read_latest()
        images.append(Image.fromarray(frame.data).convert("RGB"))
    return images


def _load_model(dtype: str, device: str):
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    processor = AutoProcessor.from_pretrained(REPO_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        REPO_ID,
        trust_remote_code=True,
        dtype=torch_dtype,
    ).to(device).eval()
    return processor, model, torch_dtype


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="SO-101 serial port, e.g. /dev/ttyACM0 or /dev/cu.usbmodem...")
    parser.add_argument("--calibration", required=True, help="PhysicalAI SO-101 calibration JSON path")
    parser.add_argument("--task", required=True, help="Language instruction for MolmoAct2")
    parser.add_argument(
        "--camera",
        action="append",
        required=True,
        help="Camera spec, repeatable. Example: top:uvc:device=0,width=640,height=480,fps=30,backend=v4l2",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run; 0 runs until Ctrl+C")
    parser.add_argument("--hz", type=float, default=2.0, help="Policy query rate. Keep low for this dirty PoC")
    parser.add_argument("--device", default="cuda", help="Torch device for policy inference")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--num-steps", type=int, default=10, help="MolmoAct2 flow solver steps")
    parser.add_argument("--actuate", action="store_true", help="Actually send predicted actions to the SO-101")
    args = parser.parse_args()

    print("Loading MolmoAct2-SO100_101...")  # noqa: T201
    processor, model, torch_dtype = _load_model(args.dtype, args.device)

    robot = SO101(port=args.port, calibration=args.calibration, role="follower")
    cameras = _connect_cameras(args.camera)
    robot.connect()

    period = 1.0 / args.hz
    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    print(f"Running {'ACTUATION' if args.actuate else 'DRY-RUN'} loop. Press Ctrl+C to stop.")  # noqa: T201

    try:
        while deadline is None or time.monotonic() < deadline:
            t0 = time.monotonic()
            obs = robot.get_observation()
            images = _read_images(cameras)
            molmo_state = _physicalai_state_to_molmo(obs.joint_positions)

            with torch.inference_mode(), torch.autocast(args.device, dtype=torch_dtype, enabled=args.dtype == "bfloat16"):
                out = model.predict_action(
                    processor=processor,
                    images=images,
                    task=args.task,
                    state=molmo_state,
                    norm_tag=NORM_TAG,
                    action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=args.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=False,
                )

            molmo_action = np.asarray(out.actions[0].detach().float().cpu().numpy(), dtype=np.float32)
            physicalai_action = _molmo_action_to_physicalai(molmo_action)
            print(  # noqa: T201
                f"state_so={np.round(molmo_state, 3).tolist()} "
                f"action_so={np.round(molmo_action, 3).tolist()} "
                f"action_rad={np.round(physicalai_action, 4).tolist()}"
            )

            if args.actuate:
                robot.send_action(physicalai_action)

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Stopping...")  # noqa: T201
    finally:
        robot.disconnect()
        for camera in cameras.values():
            camera.disconnect()


if __name__ == "__main__":
    main()
