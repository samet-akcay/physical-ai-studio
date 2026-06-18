# PhysicalAI Studio Q3 Strategy

This document aligns the `physicalai-train` and `physicalai` runtime teams for this quarter.

Companion tracking files:

- [`policies.csv`](./policies.csv)
- [`benchmark.csv`](./benchmark.csv)
- [`PhysicalAI-Studio-Feature-Improvements.md`](./PhysicalAI-Studio-Feature-Improvements.md)

Paper deadlines: **ICRA 16 Sep, ICLR 25 Sep, CVPR 14 Nov**

## Quarter Scope

The Q3 goal is to **scale and harden the existing end-to-end train-to-deploy loop for VLA and WAM policies across as many supported robots and cameras as possible**.

In scope:

- Train and benchmark first-party VLA / WAM policies.
- Export policies to deployable formats: Torch, ONNX, OpenVINO, ExecuTorch where feasible.
- Validate exported models against PyTorch behavior.
- Add quantization and edge-performance measurements.
- Expand robot and camera support.
- Add simulation environment support in the library, surfaced through Studio.
- Run exported policies through the runtime on real hardware.
- Provide a minimal one-flow path: dataset -> train -> export -> deploy.

Out of scope for Q3:

- Agentic orchestration / RobotClaw-style planner.
- Navigation as a separate VLN pillar.
- Full manipulation/navigation/world-model service separation.
- ROS2 / ZeroMQ runtime nodes.
- RL / DAgger track, except where needed to unblock WAM data design.

These are longer-term directions and are summarized at the end.

## Working Thesis

PhysicalAI Studio should be the open route from robot data to a model running on real hardware.

The repo already has most of the spine:

```text
dataset -> Observation -> train -> benchmark -> export -> manifest -> InferenceModel.load -> PolicyRuntime -> robot
```

The loop exists. This quarter is about scaling it to more SOTA policies, stronger benchmarks, more robots, and more cameras, while making the evidence measurable.

Our strongest platform differentiator is not just another policy implementation. It is the ability to take modern VLA / WAM policies and make them:

- trainable in Studio,
- exportable to deployment backends,
- validated numerically,
- quantized where useful,
- runnable in real time,
- usable across multiple robots and cameras.

## Current State

### `physicalai-train`

Already present:

- First-party policies: ACT, Pi0, Pi0.5, SmolVLA, GR00T-N1.5.
- LeRobot policy wrappers for additional LeRobot models.
- `Observation` data abstraction with numpy and torch support.
- `LeRobotDataModule` and batch-first Lightning data flow.
- Lightning-based training and jsonargparse configs.
- Benchmark runner with LIBERO and PushT support, built on the shared `Gym` abstraction.
- Export mixin for Torch, ONNX, OpenVINO, and ExecuTorch.
- Studio CLI subcommands: `fit`, `validate`, `test`, `predict`, `benchmark`, `export`.

Main gaps:

- Export support is uneven. Current backend coverage is:
  - ACT: Torch, OpenVINO, ONNX, ExecuTorch.
  - Pi0.5: Torch, OpenVINO. This is the current VLA baseline for us.
  - SmolVLA: Torch, OpenVINO.
  - Pi0 and GR00T: effectively Torch-only unless extended.
- No export-equivalence harness yet.
- Quantization is limited to FP16 OpenVINO compression; no INT8/PTQ workflow.
- No shipped WAM policy class yet.
- Backend/UI does not expose all policies consistently.

### `physicalai` Runtime

Already present:

- `InferenceModel.load(export_dir)` with manifest-driven loading.
- OpenVINO and ONNX Runtime adapters.
- Torch and ExecuTorch adapter extension point from Studio.
- `PolicyRuntime` with Sync, Async, and RTC execution.
- Manifest schema for artifacts, pre/post-processing, input/output features, robot/camera metadata.
- Robot interface and implementations for SO-101 and Trossen WidowX-AI, including bimanual Trossen.
- Camera support for UVC/V4L2, RealSense, Basler.
- iceoryx2 shared-memory camera transport.

Main gaps:

- Robot support is narrow for the target scope.
- UR / Franka extras are placeholders.
- No Unitree R1D, Oversonic, or OpenArm2 implementation yet.
- Q3 camera agenda is UVC, GMSL, MIPI, and GenICam. UVC is supported; GMSL, MIPI, and GenICam need implementation or validation.
- YAML deploy / `PolicyRuntime.from_config()` flow is not complete.
- Remote execution and richer telemetry are preview/scaffolding.

## Q3 Priorities

### 1. Scale and harden the train-to-deploy loop

Every first-party policy in this quarter's plan should be treated as incomplete until it can be exported and validated.

Definition of done for a Q3 policy:

- Config / model / policy classes are implemented.
- Training recipe is available.
- Benchmark entry is available.
- Export works for at least the planned backends.
- Export-equivalence test passes within agreed tolerance.
- Runtime loading works via `InferenceModel.load(...)`.
- At least one real-hardware deployment path is tested, where feasible.

### 2. Scale up robot and camera support

The runtime needs to run on more embodiments. Our target outcome:

- Existing robots remain stable: SO-101, Trossen WidowX-AI, bimanual Trossen.
- Generalize bimanual support where possible instead of treating bimanual Trossen as a one-off.
- Add the robot families on the Q3 agenda as hardware arrives: UR, Franka, Unitree R1D, Oversonic, and OpenArm2.
- Prioritize work by delivery date, SDK availability, and whether a safe real-eval task can be defined quickly.
- Expand camera support for the Q3 agenda: UVC, GMSL, MIPI, and GenICam.
- Decide whether non-agenda camera paths such as IP camera remain in scope this quarter.
- Each supported robot has a minimal real-eval task.

### 3. Measure what matters

The paper and internal roadmap need evidence, not just feature lists.

The following list is our required measurements:

- PyTorch vs exported backend parity.
- Latency and throughput by backend and device.
- Quantized vs non-quantized latency and accuracy.
- Sim benchmark results on non-saturated tasks.
- Simulation rollout compatibility for MuJoCo and Isaac Lab targets.
- Real-hardware success rate and runtime stability.

## Work Plan

### P0: Required for the quarter

| Item | Repo | Output | Owner |
|---|---|---|---|
| Export-equivalence harness | train | PyTorch vs Torch/ONNX/OpenVINO/ExecuTorch parity report | TBD |
| Baseline backend coverage | train | Export matrix documented; Pi0.5 baseline path hardened; Pi0/GR00T gaps scoped | TBD |
| Adopt stronger sim benchmarks | train | `vla-eval` integration for LIBERO-Pro, SimplerEnv, RoboCasa, CALVIN | TBD |
| Hardware wave 1 | runtime/backend | First available agenda robots integrated; UVC path validated; GMSL/MIPI/GenICam scoped | TBD |
| Real-hardware eval protocol | runtime/train | One frozen task per robot, common metrics, result format | TBD |
| Dataset freeze plan | train/backend | Dataset collection cutoff and model evaluation schedule | TBD |

### P1: Strong differentiators

| Item | Repo | Output | Owner |
|---|---|---|---|
| INT8/PTQ quantization | train/runtime | Calibration workflow, backend support matrix, latency/accuracy curves | TBD |
| First-party policy builds | train | 4+ VLA/WAM policies implemented and exportable | TBD |
| Hardware wave 2 | runtime/backend | Remaining agenda robots integrated; backend setup/calibration paths | TBD |
| Simulation env support | train/backend | MuJoCo adapter/benchmark path; Isaac Lab adapter design or first task | TBD |
| Policy I/O standardization | train/runtime | Consistent input/output feature schema for UI, export, runtime | TBD |
| One-flow loop UX | backend/ui/cli | dataset -> train -> export -> deploy recipe/config path | TBD |
| GR00T cleanup | train/backend | GR00T wired into backend/UI; N1.7 refactor plan or implementation | TBD |

### P2: If P0/P1 are on track

| Item | Repo | Output |
|---|---|---|
| `PolicyRuntime.from_config()` | runtime | YAML deploy path works reliably |
| Dataset quality/versioning | train/backend | Dataset versioning or quality report tooling |
| World-model data format | train | Minimal WAM data contract for DreamZero-class models |
| Isaac Lab hardening | train/backend | One selected Isaac Lab task runnable through benchmark flow, if feasible |

## Policy Implementation Plan

We expect 4+ dedicated engineers. Each engineer should own one model end-to-end.

| Engineer | Model | Class | Why | Backend target |
|---|---|---|---|---|
| E1 | MolmoAct 2 | VLA | Strong export-value model; LeRobot lacks deployment backends | ONNX + OpenVINO + ExecuTorch |
| E2 | VLA-JEPA | VLA + latent WM | Good robustness story; lower export complexity if WM is dropped at inference | ONNX + OpenVINO + Torch |
| E3 | DreamZero | WAM | Adds world-action model class; important for Q3 scope | Torch + ONNX; OpenVINO stretch |
| E4 | RLDX-1 or Qwen-RobotManip | VLA | RLDX-1 gives tactile/torque axis; Qwen-RobotManip gives direct Qwen-suite relevance | ONNX + OpenVINO |
| E5+ | GR00T N1.7, Spirit v1.5, or LingBot-VLA | VLA | Prefer finishing GR00T before adding another model if resources are limited | TBD |

Implementation order for each model:

1. Get training/parity running on a known dataset.
2. Add benchmark entry.
3. Add export path.
4. Pass export-equivalence tests.
5. Load with runtime `InferenceModel.load(...)`.
6. Run on at least one real robot, where feasible.

## Hardware Scale-Up Plan

| Type | Target | Status | Quarter target |
|---|---|---|---|
| Robot | SO-101 | Existing | Keep stable; include in eval |
| Robot | Trossen WidowX-AI | Existing | Keep stable; include in eval |
| Robot | Bimanual Trossen | Existing | Keep available; generalize bimanual handling where possible |
| Robot | Unitree R1D | Missing | Agenda target; prioritize by hardware arrival |
| Robot | UR | Placeholder/missing | Agenda target |
| Robot | Franka | Placeholder/missing | Agenda target |
| Robot | Oversonic | Missing | Agenda target |
| Robot | OpenArm2 | Missing | Agenda target |
| Robot | ABB | Placeholder | Not on Q3 agenda; remove placeholder unless owned |
| Camera | UVC/V4L2 | Existing | Baseline camera path; validate across robot setups |
| Camera | GMSL | Missing | Agenda target; define backend and hardware requirements |
| Camera | MIPI | Missing | Agenda target; define backend and hardware requirements |
| Camera | GenICam | Placeholder/missing | Agenda target; implement or validate with target camera |
| Camera | RealSense | Existing | Keep available as RGBD path, but not the main Q3 agenda |
| Camera | Basler | Existing | Keep available; likely covered through GenICam path if applicable |
| Camera | IP camera | Stub | Non-agenda; defer unless explicitly needed |

For each new robot, deliver:

- Runtime driver implementing the `Robot` protocol.
- Backend setup and calibration flow.
- UI setup path where needed.
- Minimal smoke test.
- One real-eval task.
- Documentation for connection, calibration, and safety limits.

## Benchmark Plan

Use external benchmark harnesses where possible. Build only the benchmarks that are unique to our platform.

Adopt:

- LIBERO-Pro
- SimplerEnv
- RoboCasa
- CALVIN

Build first-party:

- Export-equivalence harness.
- Real-hardware eval harness.

Track but do not reimplement this quarter:

- RoboArena
- RoboChallenge / Table30
- MolmoSpaces
- DexBench

## Simulation Environment Support

Simulation support should live in `physicalai-train` first and be surfaced by Studio. The library already has the core shape:

```text
Gym adapter -> Observation -> evaluate_policy -> Benchmark -> results/video
```

Existing pieces:

- `physicalai.gyms.Gym` defines the simulator interface: `reset`, `step`, `render`, `close`, `sample_action`, `to_observation`.
- `GymnasiumGym` already adapts Gymnasium-style environments into `Observation`.
- `Benchmark` already evaluates either a PyTorch `Policy` or exported `InferenceModel` over a list of gyms.
- LIBERO and PushT are examples of specialized benchmark wrappers around this path.

What is needed for MuJoCo:

- Add a MuJoCo/Gymnasium task adapter where generic `GymnasiumGym` is not enough.
- Normalize MuJoCo observations into `Observation.state`, `Observation.images`, and `Observation.extra`.
- Define task success extraction through `info["success"]`, `info["is_success"]`, or task-specific keys.
- Add a `MuJoCoBenchmark` wrapper that creates the task list and sets default episode counts, seeds, and max steps.
- Add optional dependency extra, for example `physicalai-train[mujoco]`.

What is needed for Isaac Lab:

- Add an Isaac Lab-specific `Gym` adapter. Do not rely on generic `GymnasiumGym` unless the target Isaac tasks cleanly expose the Gymnasium API.
- Handle vectorized GPU environments explicitly. Isaac observations/actions may already be batched and device-resident.
- Convert camera tensors, robot state, task language, rewards, and success flags into the existing `Observation` contract.
- Add an `IsaacLabBenchmark` wrapper for selected tasks, with clear install and launch requirements.
- Keep Isaac behind an optional extra, for example `physicalai-train[isaac]`, because dependencies and runtime requirements are heavy.

Studio should not own simulator semantics. Studio should:

- list available simulation environments,
- launch benchmark or rollout jobs,
- pass config to the library benchmark classes,
- store generated metrics, videos, and datasets,
- show results in the UI.

The library remains the source of truth for simulator adapters, rollout logic, benchmark protocols, and dataset conversion.

## Export and Quantization Plan

Export support should be tracked as a matrix:

```text
policy x backend x precision x device x parity status
```

Minimum expected backends:

- Torch for development and debugging.
- ONNX for portability.
- OpenVINO for Intel edge deployment.
- ExecuTorch where feasible.

Quantization scope:

- Keep FP16 OpenVINO compression.
- Add INT8/PTQ workflow, likely NNCF-based.
- Define calibration data input.
- Report latency/accuracy tradeoff.
- Do not make INT8 default until parity and quality are understood.

## One-Flow Loop

The UX goal for this quarter is not an agent. It is a simple, reproducible path from data to deployment.

Target flow:

```text
select/import dataset
-> choose policy/config
-> train
-> benchmark
-> export
-> validate exported model
-> select robot/cameras
-> deploy with runtime config
```

This can be represented first as a typed recipe/config. The UI can then wrap the same recipe as a guided flow.

## Evidence for the Paper

| Claim | Evidence |
|---|---|
| Export is reliable | PyTorch vs exported backend parity table |
| Runtime is practical | Latency/throughput on CPU/GPU/NPU/CUDA where available |
| Edge deployment is useful | FP16/INT8 latency and quality tradeoff |
| Policies are not only LIBERO demos | LIBERO-Pro, SimplerEnv, RoboCasa, CALVIN results |
| Simulation support is reusable | MuJoCo/Isaac Lab tasks run through the same `Gym` -> `Benchmark` path |
| Hardware support is real | Success rate and stability across 3-4 robot embodiments |
| Loop is scaled | End-to-end Pi0.5 baseline plus new SOTA VLA/WAM policies: data -> train -> export -> deploy |

## Open Decisions

- E4 model: RLDX-1 or Qwen-RobotManip?
- Hardware sequencing: which agenda robots arrive first, and who owns each driver?
- ABB: remove placeholder unless someone owns it this quarter?
- Camera sequencing: which GMSL/MIPI/GenICam devices arrive first, and who owns each path?
- Simulation sequencing: MuJoCo first and Isaac Lab second, or parallel owners?
- Minimum robot count for paper table: 3 or 4 embodiments?
- DreamZero/WAM data format: can it fit `LeRobotDataModule`, or do we need a new format?
- Owners for each P0 item.

## Longer-Term Direction

Qwen-Robot Suite is a useful reference for the longer-term architecture: manipulation, navigation, and world models as separate capabilities, composed by an agent layer.

That is not the Q3 deliverable. The Q3 deliverable is scaling the existing lower-level foundation that makes such a system credible: more policies that can be trained, exported, validated, quantized, and deployed on real robots.

After Q3, likely next steps are:

- Agentic orchestration layer where a planner calls exported policies as tools.
- Navigation/VLN policy class and nav-sim integration.
- World model used as synthetic data engine and evaluator.
- RL / DAgger data formats and online learning support.
- Cross-embodiment canonical state/action representation.
- ROS2 / ZeroMQ nodes and remote inference execution.

The sequencing is intentional: **scale and harden train-to-deploy first; compose capabilities after that.**
