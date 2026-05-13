# MolmoAct2 SO101 Handoff

## Branches
- PR base work is on `feature/lerobot-torch-inference-parity`
- PR: `https://github.com/open-edge-platform/physical-ai-studio/pull/585`
- Molmo follow-on work is on `feature/molmoact2-so101-poc`

## What changed on the Molmo branch
- Added optional MolmoAct2 policy wrapper under `src/physicalai/policies/molmoact2/`
- Exported `MolmoAct2` from `src/physicalai/policies/__init__.py`
- Added lightweight MolmoAct2 tests in `tests/unit/policies/test_molmoact2.py`
- Added LeRobot helper import compatibility in `src/physicalai/policies/lerobot/policy.py`
- Added generic `Observation.state` flattening support in `src/physicalai/data/lerobot/converters.py`
  - `state=tensor` -> `observation.state`
  - `state={"state": tensor}` -> `observation.state`
  - `state={"joint": ..., "gripper": ...}` -> `observation.state.joint`, `observation.state.gripper`
- Added regression tests in `tests/unit/data/test_observation.py`
- Added an SO101 PoC script at `examples/so101/molmoact2_dry_run.py`

## Verified model checkpoints
- `allenai/MolmoAct2-LIBERO`
  - loads through the wrapper
  - smoke inference works
- `allenai/MolmoAct2-SO100_101`
  - loads through the wrapper
  - requires `norm_tag="so100_so101_molmoact2"`
  - smoke inference works
  - returns action shape `(1, 6)`

## Important state/action convention detail
PhysicalAI SO101 and LeRobot SO100/101 do not use the same control units by default.

- PhysicalAI `SO101` driver returns calibrated joint positions in radians
- LeRobot SO100/101 defaults to degrees for the first 5 joints and `0-100` for gripper
- MolmoAct2 SO100/101 follows the LeRobot SO convention

The PoC script explicitly converts:
- PhysicalAI radians -> MolmoAct2 SO state convention before inference
- MolmoAct2 predicted action -> PhysicalAI radians before optional actuation

## Recommended hardware topology
Use the Linux workstation next to the robot, not the remote Linux server in this chat.

Recommended:
- connect SO101 and cameras to the Linux workstation with the 3090s
- run the PoC there
- use this remote server only for code sync, commits, and handoff notes

Fallback:
- if the robot must stay attached to the MacBook, run robot/camera IO on the MacBook and remote inference from the Linux workstation

Do not put the remote Linux server in the realtime robot control loop.

## PoC script
Path:
- `examples/so101/molmoact2_dry_run.py`

Purpose:
- uses `physicalai` robot and camera APIs for deployment-side IO
- loads `allenai/MolmoAct2-SO100_101` directly via Transformers
- runs dry-run by default
- only actuates if `--actuate` is explicitly passed

Example command:

```bash
uv run python examples/so101/molmoact2_dry_run.py \
  --port /dev/ttyACM0 \
  --calibration path/to/so101_calibration.json \
  --task "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl." \
  --camera top:uvc:device=0,width=640,height=480,fps=30,backend=v4l2 \
  --camera side:uvc:device=2,width=640,height=480,fps=30,backend=v4l2 \
  --device cuda \
  --dtype bfloat16 \
  --duration 30
```

Only after inspection, enable real actuation with:

```bash
--actuate
```

## Runtime design relation
The target architecture is in:
- `docs/design/components/runtime-system/robot_runtime_architecture.md`

Current PoC status:
- not a full `RobotRuntime + PolicyController` implementation yet
- just a narrow proof-of-concept loop that exercises the same core data path
  - read robot state
  - read cameras
  - run policy
  - optionally send action

This is intended as a stepping stone toward the real runtime design.

## Verification already run
- `uv run pytest tests/unit/data/test_observation.py tests/unit/policies/test_molmoact2.py tests/unit/policies/test_get_policy.py tests/unit/policies/test_lerobot.py::TestNamedLeRobotPolicy::test_named_wrappers_share_policy_name_with_base -q`
- real MolmoAct2 smoke with `allenai/MolmoAct2-LIBERO`
- real MolmoAct2 smoke with `allenai/MolmoAct2-SO100_101`
- `uv run python -m py_compile examples/so101/molmoact2_dry_run.py`

## Open issues / caveats
- `lerobot-rollout --policy.path=allenai/MolmoAct2-SO100_101` does not work directly because the HF repo is a raw Transformers checkpoint, not a LeRobot `PreTrainedConfig` repo with top-level `type`
- async LeRobot remote inference path currently does not list `molmoact2` in its fixed supported policy list
- the PoC script depends on the sibling `physicalai` package being available via the workspace layout in this repo

## Suggested next steps on the workstation
1. Pull `feature/molmoact2-so101-poc`
2. Install the needed local hardware deps
3. Run `examples/so101/molmoact2_dry_run.py` without `--actuate`
4. Inspect states/actions and confirm calibration
5. Only then try `--actuate`
6. If the path works, refactor the PoC toward `RobotRuntime + PolicyController`
