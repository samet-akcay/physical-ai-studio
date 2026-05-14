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
from physicalai.robot.so101.calibration import SO101Calibration
from physicalai.robot.so101.constants import RADIANS_PER_TICK, SO101_JOINT_ORDER, TICKS_PER_REVOLUTION


REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

# LeRobot SO follower normalization (lerobot/motors/motors_bus.py:_normalize, lines 858-861)
#   DEGREES: deg = (raw_present_position - mid) * 360 / (max_res - 1)
#
# In LeRobot's calibration regime (set_half_turn_homings, motors_bus.py:762), the
# servo's Homing_Offset register is written so that the calibrated zero pose reads
# Present_Position == 2047 (half turn on a 12-bit encoder). Range_min/range_max
# are then the user-recorded extremes of motion. The midpoint of those recorded
# ranges only equals the half-turn (~2047) if the robot's mechanical workspace is
# symmetric about the calibrated zero -- which is NOT true for SO-101 shoulder_lift
# or wrist_roll. Using the asymmetric (range_min+range_max)/2 from a non-LeRobot
# calibration produces a frame offset that pushes those joints far out of the
# MolmoAct2 SO100/101 training distribution.
#
# To match LeRobot training data, we offer two midpoint modes:
#   - "half-turn": mid = (max_res - 1) / 2 = 2047.5  (LeRobot-faithful)
#   - "calib":    mid = (range_min + range_max) / 2 (legacy PhysicalAI behavior)
_MAX_RES = TICKS_PER_REVOLUTION  # 4096
_HALF_TURN = (_MAX_RES - 1) / 2.0  # 2047.5


def _physicalai_rad_to_raw_tick(rad: float, calibration: SO101Calibration, joint_name: str) -> float:
    """Invert PhysicalAI's tick→rad to recover the raw servo tick (float, no rounding)."""
    cal = calibration.joints[joint_name]
    return rad / (cal.direction * RADIANS_PER_TICK) + cal.homing_offset


def _midpoint(calibration: SO101Calibration, joint_name: str, mode: str) -> float:
    if mode == "half-turn":
        return _HALF_TURN
    if mode == "calib":
        cal = calibration.joints[joint_name]
        return (cal.range_min + cal.range_max) / 2.0
    msg = f"Unknown midpoint mode {mode!r}; expected 'half-turn' or 'calib'"
    raise ValueError(msg)


def _raw_tick_to_lerobot(
    raw_tick: float,
    calibration: SO101Calibration,
    joint_name: str,
    midpoint_mode: str,
) -> float:
    """Apply LeRobot SO follower _normalize for a single joint (DEGREES mode)."""
    mid = _midpoint(calibration, joint_name, midpoint_mode)
    return (raw_tick - mid) * 360.0 / (_MAX_RES - 1)


def _lerobot_to_raw_tick(
    lerobot_value: float,
    calibration: SO101Calibration,
    joint_name: str,
    midpoint_mode: str,
) -> float:
    """Inverse of _raw_tick_to_lerobot (LeRobot _unnormalize, DEGREES mode)."""
    mid = _midpoint(calibration, joint_name, midpoint_mode)
    return lerobot_value * (_MAX_RES - 1) / 360.0 + mid


def _raw_tick_to_physicalai_rad(raw_tick: float, calibration: SO101Calibration, joint_name: str) -> float:
    """Apply PhysicalAI's tick→rad formula (so101.py:_ticks_to_radians) for one joint."""
    cal = calibration.joints[joint_name]
    return (raw_tick - cal.homing_offset) * cal.direction * RADIANS_PER_TICK


def _physicalai_state_to_molmo(
    state_rad: np.ndarray,
    calibration: SO101Calibration,
    midpoint_mode: str,
) -> np.ndarray:
    """Convert PhysicalAI SO101 radians to LeRobot SO100/101 DEGREES frame."""
    state_rad = np.asarray(state_rad, dtype=np.float32)
    out = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
    for i, name in enumerate(SO101_JOINT_ORDER):
        raw_tick = _physicalai_rad_to_raw_tick(float(state_rad[i]), calibration, name)
        out[i] = _raw_tick_to_lerobot(raw_tick, calibration, name, midpoint_mode)
    return out


def _molmo_action_to_physicalai(
    action: np.ndarray,
    calibration: SO101Calibration,
    midpoint_mode: str,
) -> np.ndarray:
    """Convert MolmoAct2 SO100/101 action (DEGREES) to PhysicalAI radians."""
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        return _convert_single_action(action, calibration, midpoint_mode)
    if action.ndim == 2:
        return np.stack([_convert_single_action(row, calibration, midpoint_mode) for row in action], axis=0)
    msg = f"Unexpected action ndim {action.ndim}; expected 1 or 2"
    raise ValueError(msg)


def _convert_single_action(
    action: np.ndarray,
    calibration: SO101Calibration,
    midpoint_mode: str,
) -> np.ndarray:
    out = np.empty(len(SO101_JOINT_ORDER), dtype=np.float32)
    for i, name in enumerate(SO101_JOINT_ORDER):
        raw_tick = _lerobot_to_raw_tick(float(action[i]), calibration, name, midpoint_mode)
        out[i] = _raw_tick_to_physicalai_rad(raw_tick, calibration, name)
    return out


def _safety_clamp(
    target: np.ndarray,
    current: np.ndarray,
    max_delta_rad: float,
    max_gripper_delta: float,
) -> np.ndarray:
    """Clamp target action to ``current ± max_delta`` per joint.

    Args:
        target: Desired joint positions, shape ``(6,)``, radians for joints
            0-4 and raw gripper units for joint 5.
        current: Currently observed joint positions, same shape/units as
            ``target``.
        max_delta_rad: Maximum allowed change per cycle for the 5 arm joints,
            in radians.
        max_gripper_delta: Maximum allowed change per cycle for the gripper.

    Returns:
        Clamped target with the same shape as ``target``.
    """
    clamped = target.copy()
    deltas = np.array([max_delta_rad] * 5 + [max_gripper_delta], dtype=np.float32)
    lower = current - deltas
    upper = current + deltas
    return np.clip(clamped, lower, upper).astype(np.float32)


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


def _goto_start_pose(
    robot: SO101,
    calibration: SO101Calibration,
    midpoint_mode: str,
    max_step_deg: float = 5.0,
    hz: float = 10.0,
    tolerance_deg: float = 2.0,
    skip_joints: list[int] | None = None,
) -> None:
    """Move the arm slowly to the MolmoAct2 SO100/101 training-distribution median.

    Uses the q50 (median) from norm_stats.json state_stats as the target. Moves at
    most ``max_step_deg`` per cycle on each joint, at ``hz`` control rate, until all
    joints are within ``tolerance_deg`` of the target.
    """
    # q50 from norm_stats.json metadata_by_tag/so100_so101_molmoact2/state_stats
    target_lerobot_deg = np.array([3.07, 123.16, 124.40, 57.89, -11.04, 9.24], dtype=np.float32)

    target_rad = _molmo_action_to_physicalai(target_lerobot_deg, calibration, midpoint_mode)
    max_step_rad = float(np.deg2rad(max_step_deg))
    period = 1.0 / hz
    _skip = set(skip_joints or [])
    if _skip:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        print(f"Skipping joints: {[joint_names[i] for i in sorted(_skip)]}")  # noqa: T201
    active_mask = np.array([i not in _skip for i in range(6)])

    # Check initial error — warn if arm is far from target (servos may stall under gravity)
    obs_init = robot.get_observation()
    init_deg = _physicalai_state_to_molmo(obs_init.joint_positions, calibration, midpoint_mode)
    init_error = np.abs(target_lerobot_deg - init_deg)
    if init_error.max() > 90.0:
        print(  # noqa: T201
            f"WARNING: Arm is very far from target (max error {init_error.max():.0f}°).\n"
            f"Servos may stall under gravity. If movement stalls, manually\n"
            f"reposition the arm closer to the target pose and re-run.\n"
        )

    print(  # noqa: T201
        f"Moving to start pose (training median): {np.round(target_lerobot_deg, 1).tolist()} deg (LeRobot frame)\n"
        f"Speed: {max_step_deg} deg/step at {hz} Hz. Ctrl+C to abort."
    )

    try:
        step = 0
        stall_count = 0
        prev_err_max = None
        read_errors = 0
        while True:
            t0 = time.monotonic()
            try:
                obs = robot.get_observation()
            except ConnectionError:
                read_errors += 1
                if read_errors > 10:
                    print(  # noqa: T201
                        "\n*** BUS ERROR: Servo communication repeatedly failing. ***\n"
                        "Possible causes: servo thermal protection, loose cable, power issue.\n"
                        "Try: power-cycle the arm (unplug/replug USB) and re-run.\n"
                    )
                    return
                time.sleep(0.1)
                continue
            read_errors = 0
            current = obs.joint_positions
            current_deg = _physicalai_state_to_molmo(current, calibration, midpoint_mode)

            error_deg = np.abs(target_lerobot_deg - current_deg)
            active_error = error_deg[active_mask]
            if np.all(active_error < tolerance_deg):
                print(f"Reached start pose. Final state (LeRobot deg): {np.round(current_deg, 2).tolist()}")  # noqa: T201
                return

            # Detect stall: if error hasn't decreased in 50 steps, servos are stuck
            err_max = float(active_error.max())
            if prev_err_max is not None and abs(err_max - prev_err_max) < 0.5:
                stall_count += 1
            else:
                stall_count = 0
            prev_err_max = err_max

            if stall_count >= 50:
                worst_joint = int(np.argmax(error_deg))
                joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                print(  # noqa: T201
                    f"\n*** STALL DETECTED after {step} steps ***\n"
                    f"Servos cannot reach target from current position.\n"
                    f"Worst joint: {joint_names[worst_joint]} (err={error_deg[worst_joint]:.1f}°)\n"
                    f"Current: {np.round(current_deg, 1).tolist()}\n"
                    f"Target:  {np.round(target_lerobot_deg, 1).tolist()}\n\n"
                    f"FIX: Manually position the arm closer to the target pose,\n"
                    f"then re-run. The shoulder_lift and elbow should be roughly upright.\n"
                )
                return

            clamped = _safety_clamp(
                target_rad,
                current,
                max_delta_rad=max_step_rad,
                max_gripper_delta=max_step_rad,
            )
            for j in _skip:
                clamped[j] = current[j]
            robot.send_action(clamped)

            if step % 10 == 0:
                print(  # noqa: T201
                    f"  step={step:4d} err_max={error_deg.max():.1f}° "
                    f"state_deg={np.round(current_deg, 1).tolist()} "
                    f"target_deg={np.round(target_lerobot_deg, 1).tolist()}"
                )
            step += 1

            elapsed = time.monotonic() - t0
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("Start-pose movement aborted by user.")  # noqa: T201
        raise SystemExit(1)  # noqa: B904


def _run_diagnose(robot: SO101, calibration: SO101Calibration) -> None:
    """One-shot frame diagnostic: read robot once and print all reference frames.

    Compares both midpoint conventions against the MolmoAct2 SO100/101 training
    distribution (q01..q99 from norm_stats.json metadata_by_tag/so100_so101_molmoact2)
    so the operator can pick the convention that lands in-distribution.

    Also reads the firmware Homing_Offset register (addr 31, 2 bytes) from each
    servo to reveal whether LeRobot calibration was previously applied to the
    hardware. If nonzero, the Present_Position the driver reads is already
    shifted, and the correct LeRobot degree is:
        deg = (present_pos - 2047.5) * 360 / 4095
    which is exactly what 'half-turn' mode computes (since raw_tick IS present_pos).
    """
    obs = robot.get_observation()
    state_rad = np.asarray(obs.joint_positions, dtype=np.float64)

    raw_ticks = np.array(
        [_physicalai_rad_to_raw_tick(float(state_rad[i]), calibration, name)
         for i, name in enumerate(SO101_JOINT_ORDER)],
        dtype=np.float64,
    )
    deg_half = np.array(
        [_raw_tick_to_lerobot(float(raw_ticks[i]), calibration, name, "half-turn")
         for i, name in enumerate(SO101_JOINT_ORDER)],
        dtype=np.float64,
    )
    deg_calib = np.array(
        [_raw_tick_to_lerobot(float(raw_ticks[i]), calibration, name, "calib")
         for i, name in enumerate(SO101_JOINT_ORDER)],
        dtype=np.float64,
    )

    # Read firmware Homing_Offset (register 31, 2 bytes, signed) from each servo
    firmware_offsets: dict[str, int] = {}
    conn = robot._require_connection()  # noqa: SLF001
    for name in SO101_JOINT_ORDER:
        servo_id = robot.servo_ids[name]
        val, comm, err = conn.packet_handler.read2ByteTxRx(conn.port_handler, servo_id, 31)
        if comm != 0 or err != 0:
            firmware_offsets[name] = -9999
        else:
            # STS3215 Homing_Offset is signed 16-bit
            firmware_offsets[name] = val if val < 32768 else val - 65536

    state_q01 = (-41.91, 43.67, 38.39, 5.71, -63.45, 0.94)
    state_q99 = (48.29, 185.26, 173.14, 91.78, 42.94, 44.14)

    def _in(value: float, lo: float, hi: float) -> str:
        return "ok" if lo <= value <= hi else "OOD"

    print("MolmoAct2 SO100/101 frame diagnostic (single sample)")  # noqa: T201
    print(f"{'joint':<14} {'tick':>7} {'fw_off':>7} {'rad':>9} {'deg(half)':>11} "  # noqa: T201
          f"{'fit(half)':>10} {'deg(calib)':>11} {'fit(calib)':>11} "
          f"{'train q01..q99':>20}")
    for i, name in enumerate(SO101_JOINT_ORDER):
        cal = calibration.joints[name]
        calib_mid = (cal.range_min + cal.range_max) / 2.0
        fw = firmware_offsets[name]
        print(  # noqa: T201
            f"{name:<14} {raw_ticks[i]:7.0f} {fw:7d} {state_rad[i]:9.4f} "
            f"{deg_half[i]:11.2f} {_in(deg_half[i], state_q01[i], state_q99[i]):>10} "
            f"{deg_calib[i]:11.2f} {_in(deg_calib[i], state_q01[i], state_q99[i]):>11} "
            f"{state_q01[i]:>8.2f}..{state_q99[i]:<8.2f} "
            f"(calib_mid={calib_mid:.0f}, sw_homing={cal.homing_offset})"
        )

    print("\nFirmware Homing_Offset interpretation:")  # noqa: T201
    any_nonzero = any(v != 0 and v != -9999 for v in firmware_offsets.values())
    if any_nonzero:
        print("  Firmware offsets are NONZERO — LeRobot calibration was previously applied.")  # noqa: T201
        print("  Present_Position is already homed. 'half-turn' mode should be correct.")  # noqa: T201
    else:
        all_read_ok = all(v != -9999 for v in firmware_offsets.values())
        if all_read_ok:
            print("  Firmware offsets are ALL ZERO — factory defaults. Present_Position = Actual_Position.")  # noqa: T201
            print("  The servo has NOT been calibrated via LeRobot's set_half_turn_homings.")  # noqa: T201
            print("  To match MolmoAct2 training, either:")  # noqa: T201
            print("    (a) run LeRobot calibration (writes firmware offset), or")  # noqa: T201
            print("    (b) manually apply the PhysicalAI homing_offset as a firmware-style correction:")  # noqa: T201
            print("        corrected_tick = raw_tick + sw_homing_offset - 2047")  # noqa: T201
            print("        deg = (corrected_tick - 2047.5) * 360 / 4095")  # noqa: T201
        else:
            print("  Could not read firmware offset from some servos.")  # noqa: T201

    print(  # noqa: T201
        "\nPick --midpoint=half-turn if the 'fit(half)' column is mostly 'ok'. "
        "Pick --midpoint=calib if 'fit(calib)' is mostly 'ok'. "
        "If neither, the calibration convention doesn't match LeRobot training — "
        "see guidance above."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="SO-101 serial port, e.g. /dev/ttyACM0 or /dev/cu.usbmodem...")
    parser.add_argument("--calibration", required=True, help="PhysicalAI SO-101 calibration JSON path")
    parser.add_argument("--task", help="Language instruction for MolmoAct2 (required unless --diagnose)")
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera spec, repeatable. Required unless --diagnose. "
             "Example: top:uvc:device=0,width=640,height=480,fps=30,backend=v4l2",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run; 0 runs until Ctrl+C")
    parser.add_argument("--hz", type=float, default=2.0, help="Policy query rate. Keep low for this dirty PoC")
    parser.add_argument("--device", default="cuda", help="Torch device for policy inference")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--num-steps", type=int, default=10, help="MolmoAct2 flow solver steps")
    parser.add_argument("--actuate", action="store_true", help="Actually send predicted actions to the SO-101")
    parser.add_argument(
        "--max-delta-deg",
        type=float,
        default=8.0,
        help="Safety clamp: max change per cycle for arm joints 0-4 (degrees). Used only with --actuate.",
    )
    parser.add_argument(
        "--max-gripper-delta-rad",
        type=float,
        default=0.3,
        help="Safety clamp: max gripper change per cycle in radians (PhysicalAI gripper is in rad). Used with --actuate.",
    )
    parser.add_argument(
        "--midpoint",
        choices=("half-turn", "calib"),
        default="half-turn",
        help=(
            "Midpoint used by the LeRobot DEGREES conversion. 'half-turn' uses 2047.5 "
            "(LeRobot-faithful, matches MolmoAct2 SO100/101 training). 'calib' uses "
            "(range_min+range_max)/2 from the PhysicalAI calibration JSON (legacy)."
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Read robot once, print raw ticks + radians + both LeRobot frames, then exit. Skips inference.",
    )
    parser.add_argument(
        "--goto-start-pose",
        action="store_true",
        help="Slowly move the arm to the MolmoAct2 training-distribution median pose before inference.",
    )
    parser.add_argument(
        "--start-pose-speed-deg",
        type=float,
        default=5.0,
        help="Max degrees per step when moving to start pose (lower = slower/safer).",
    )
    parser.add_argument(
        "--skip-joints",
        type=str,
        nargs="*",
        default=[],
        help="Joint indices (0-5) to skip during goto-start-pose (e.g. --skip-joints 1 for shoulder_lift).",
    )
    args = parser.parse_args()
    args.skip_joints = [int(j) for j in args.skip_joints]

    calibration = SO101Calibration.from_path(args.calibration)
    robot = SO101(port=args.port, calibration=args.calibration, role="follower")
    robot.connect()
    try:
        if args.diagnose:
            _run_diagnose(robot, calibration)
            return
    finally:
        if args.diagnose:
            robot.disconnect()

    if args.goto_start_pose:
        _goto_start_pose(robot, calibration, args.midpoint, max_step_deg=args.start_pose_speed_deg, skip_joints=args.skip_joints)
        if not args.task or not args.camera:
            print("Start pose reached. Exiting (no --task/--camera for inference).")  # noqa: T201
            robot.disconnect()
            return

    if not args.task or not args.camera:
        msg = "--task and --camera are required unless --diagnose or --goto-start-pose is set"
        raise SystemExit(msg)

    print("Loading MolmoAct2-SO100_101...")  # noqa: T201
    processor, model, torch_dtype = _load_model(args.dtype, args.device)

    cameras = _connect_cameras(args.camera)

    period = 1.0 / args.hz
    deadline = None if args.duration == 0 else time.monotonic() + args.duration
    print(f"Running {'ACTUATION' if args.actuate else 'DRY-RUN'} loop. Press Ctrl+C to stop.")  # noqa: T201

    try:
        while deadline is None or time.monotonic() < deadline:
            t0 = time.monotonic()
            obs = robot.get_observation()
            images = _read_images(cameras)
            molmo_state = _physicalai_state_to_molmo(obs.joint_positions, calibration, args.midpoint)

            with torch.inference_mode(), torch.autocast(args.device, dtype=torch_dtype, enabled=args.dtype == "bfloat16"):
                out = model.predict_action(
                    processor=processor,
                    images=images,
                    task=args.task,
                    state=molmo_state,
                    norm_tag=NORM_TAG,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=args.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=False,
                )

            molmo_action_chunk = np.asarray(out.actions[0].detach().float().cpu().numpy(), dtype=np.float32)
            molmo_action_step0 = molmo_action_chunk[0]
            physicalai_action = _molmo_action_to_physicalai(molmo_action_step0, calibration, args.midpoint)

            if args.actuate:
                max_delta_rad = float(np.deg2rad(args.max_delta_deg))
                clamped_action = _safety_clamp(
                    physicalai_action,
                    obs.joint_positions,
                    max_delta_rad=max_delta_rad,
                    max_gripper_delta=args.max_gripper_delta_rad,
                )
                print(  # noqa: T201
                    f"chunk_shape={molmo_action_chunk.shape} "
                    f"state_so={np.round(molmo_state, 3).tolist()} "
                    f"action_so={np.round(molmo_action_step0, 3).tolist()} "
                    f"action_rad={np.round(physicalai_action, 4).tolist()} "
                    f"clamped_rad={np.round(clamped_action, 4).tolist()}"
                )
                robot.send_action(clamped_action)
            else:
                print(  # noqa: T201
                    f"chunk_shape={molmo_action_chunk.shape} "
                    f"state_so={np.round(molmo_state, 3).tolist()} "
                    f"action_so={np.round(molmo_action_step0, 3).tolist()} "
                    f"action_rad={np.round(physicalai_action, 4).tolist()}"
                )

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
