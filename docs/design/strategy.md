# Strategy: Geti Action Architecture

## Executive Summary

Geti Action is an end-to-end platform for robot AI development: data collection, training, and deployment. It consists of two packages — a **library** and an **application** — with clear ownership boundaries.

**Two core architectural decisions:**

1. **Library-first**: The library owns every component needed for end-to-end robot AI (robots, cameras, teleop, data collection, policies, inference, training, export). The application is purely UI and orchestration — glue on top.

2. **Three-layer deployment stack**: For inference/deployment, we propose a clean layering — **inferencekit** (generic inference) → **getiaction** (robotics inference) → **physical‑ai‑framework** (deployment shell) — so deployment can work independently of the full application.

---

## Part 1: Library-First Architecture

### The Principle

The **library** (`pip install getiaction`) is the single source of truth for all core components. The **application** (Studio) is a UI/orchestration layer that imports and composes library components — it adds no core logic of its own.

This means:

- A researcher can do everything via Python scripts or CLI — no web server needed.
- The application gets capabilities for free as the library evolves.
- Edge deployment requires only the library, not the full application stack.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        APPLICATION (Studio)                         │
│                                                                     │
│   UI (React)  │  Backend (FastAPI)  │  Workflows  │  Calibration    │
│               │                     │             │                 │
│   Orchestrates and composes library components — adds no core logic │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ imports
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        LIBRARY (getiaction)                         │
│                                                                     │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────────────┐   │
│  │  Robots   │ │  Cameras  │ │  Teleop   │ │  Data Collection   │   │
│  │  ABC +    │ │  ABC +    │ │  Session  │ │  DatasetWriter +   │   │
│  │  drivers  │ │  drivers  │ │  manager  │ │  episode mgmt      │   │
│  └───────────┘ └───────────┘ └───────────┘ └────────────────────┘   │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌────────────────────┐   │
│  │ Policies  │ │ Inference │ │ Training  │ │  Export            │   │
│  │  ACT, Pi0 │ │  via      │ │  Lightning│ │  ONNX, OpenVINO    │   │
│  │  SmolVLA  │ │  infkit   │ │  trainer  │ │  TorchExport       │   │
│  └───────────┘ └───────────┘ └───────────┘ └────────────────────┘   │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                          │
│  │    CLI    │ │   Eval    │ │  Config   │                          │
│  │  getiact  │ │  rollout  │ │  jsonarg  │                          │
│  │  commands │ │  metrics  │ │  parse    │                          │
│  └───────────┘ └───────────┘ └───────────┘                          │
└─────────────────────────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
   Robot SDKs      Camera SDKs    LeRobot datasets
 (optional extras) (optional extras)  + HF Hub
```

### Ownership Boundary

| Component                           | Library | Application |
| ----------------------------------- | :-----: | :---------: |
| Robot ABC + SDK drivers             |    ✓    |   imports   |
| Camera ABC + SDK drivers            |    ✓    |   imports   |
| Teleoperation sessions              |    ✓    |   imports   |
| Data collection / episodes          |    ✓    |   imports   |
| Policies (ACT, Pi0, SmolVLA, GR00T) |    ✓    |      —      |
| Inference engine                    |    ✓    |    uses     |
| Training (Lightning)                |    ✓    |    uses     |
| Export (ONNX, OpenVINO)             |    ✓    |      —      |
| Evaluation / rollouts               |    ✓    |      —      |
| CLI (`getiaction`)                  |    ✓    |      —      |
| Calibration                         |    —    |      ✓      |
| GUI / web UI                        |    —    |      ✓      |
| Workflow orchestration              |    —    |      ✓      |

### Why Library-First?

| Benefit                    | Explanation                                                     |
| -------------------------- | --------------------------------------------------------------- |
| **Edge deployment**        | `pip install getiaction[inference]` + a script. No web server.  |
| **Consistent workflows**   | Same Python API whether called from CLI, script, or application |
| **Faster adoption**        | Researchers start with `pip install`, not a full stack          |
| **Single source of truth** | No version skew between library and application                 |
| **Testable in isolation**  | Every component has unit tests without application dependencies |

### Current State vs Target

Today, robot/camera drivers live in the **application** backend (`application/backend/src/robots/`, `application/backend/src/workers/`). The target state moves these into the **library** so they're available everywhere:

```
Current                              Target
─────────────────────                ──────────────────────
application/                         application/
  robots/ ← robot drivers              (imports from library)
  workers/ ← camera, teleop            (imports from library)
                                     library/
library/                               robots/     ← moved here
  policies/                            cameras/    ← moved here
  inference/                           teleop/     ← moved here
  data/                                data/       ← already here
  ...                                  policies/   ← already here
                                       inference/  ← already here
                                       ...
```

### Modular Installation

Robot and camera SDKs are optional extras — the core install stays lightweight.

```bash
pip install getiaction                # Core API only (no hardware SDKs)
pip install getiaction[lerobot]       # LeRobot dependencies
pip install getiaction[trossen]       # Trossen robots
pip install getiaction[ur]            # Universal Robots
pip install getiaction[robots]        # All robot SDKs
pip install getiaction[realsense]     # Intel RealSense cameras
pip install getiaction[cameras]       # All camera SDKs
```

---

## Part 2: Deployment Stack

### The Problem

After data collection and training, you have an exported model. How do you run it on a robot? The deployment stack answers this with a clean three-layer architecture.

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              physical‑ai‑framework (shell)                  │
│                                                             │
│  Thin deployment platform. CLI + packaging + docs.          │
│  Can use getiaction, LeRobot, or other frameworks.          │
│                                                             │
│  pip install physical-ai-framework                          │
│  phyai run --config deploy.yaml                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ depends on
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    getiaction (library)                     │
│                                                             │
│  Robotics-specific inference: policies, robot control,      │
│  action chunking, preprocessing, LeRobot compatibility.     │
│                                                             │
│  from getiaction.inference import PolicyRunner              │
│  runner = PolicyRunner.load("./exports/act_policy")         │
└──────────────────────────┬──────────────────────────────────┘
                           │ depends on
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    inferencekit (core)                      │
│                                                             │
│  Domain-agnostic inference: InferenceModel, RuntimeAdapter, │
│  backend abstraction (OpenVINO, ONNX, TensorRT, Torch).     │
│  No robotics, no cameras, no domain logic.                  │
│                                                             │
│  from inferencekit import InferenceModel                    │
│  model = InferenceModel.load("./exports/my_model")          │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer                     | Owns                                                                                                        | Does NOT own                              |
| ------------------------- | ----------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **inferencekit**          | Model loading, backend adapters (OpenVINO/ONNX/TensorRT/Torch), runners, callbacks, metadata                | Robotics, cameras, policies, domain logic |
| **getiaction**            | Robot/camera/teleop APIs, policies, preprocessing, action chunking, LeRobot compat, domain-specific runners | Deployment CLI, packaging, branding       |
| **physical‑ai‑framework** | CLI entrypoints (`phyai run`, `phyai serve`), packaging, deployment docs, production config                 | Core implementations (thin wrappers only) |

### Why Three Layers?

1. **inferencekit** is reusable beyond robotics (computer vision, NLP, anomaly detection). Keeping it domain-agnostic maximizes reuse.
2. **getiaction** adds robotics intelligence on top of generic inference. This is where action chunking, policy runners, and hardware control live.
3. **physical‑ai‑framework** is the product boundary — a lightweight shell that leadership can position and market independently, without duplicating engineering.

### Key Rule

Dependencies flow **upward only**:

```
physical‑ai‑framework → getiaction[inference] → inferencekit
```

physical‑ai‑framework depends on getiaction. getiaction depends on inferencekit. Never the reverse.

### When to Split physical‑ai‑framework

Split into its own core package **only if**:

- There are **2+ external consumers** beyond getiaction
- You need **independent release cadences**
- You can commit to **contract tests + versioning discipline**

Until then, a thin shell provides the product boundary without engineering risk.

---

## User-Facing Experience

### Data Collection (Library)

```python
from getiaction.robots import SO101
from getiaction.teleop import TeleopSession
from getiaction.data import DatasetWriter

robot = SO101.from_config("robot.yaml")
with TeleopSession(robot=robot) as teleop:
    with DatasetWriter(path="./dataset", robot=robot, format="lerobot") as writer:
        for step in teleop:
            writer.add_frame(step.observation, action=step.action)
        writer.save_episode(task="demo")
        writer.finalize()
```

### CLI — Data Collection

```bash
getiaction teleop --config teleop.yaml
getiaction record --config record.yaml
getiaction upload --repo-id org/my-dataset --path ./dataset
```

### CLI — Deployment (physical‑ai‑framework)

```bash
phyai run --config deploy.yaml
phyai serve --model ./exports/policy --robot robot.yaml
```

---

## Key Decisions

| Decision              | Choice                                | Rationale                                 |
| --------------------- | ------------------------------------- | ----------------------------------------- |
| Core logic location   | getiaction library                    | Single source of truth, no version skew   |
| Application role      | UI + orchestration only               | Library-first; app adds no core logic     |
| Deployment packaging  | Thin shell (physical‑ai‑framework)    | Product boundary without engineering risk |
| Robot/camera API home | Library, not application              | Edge deployment without web server        |
| Installation model    | Optional extras per SDK               | Lightweight core, no SDK pollution        |
| Async strategy        | Async core + sync wrapper             | App needs async, CLI needs sync           |
| CLI framework         | LightningCLI / jsonargparse           | Align with existing getiaction patterns   |
| Dataset format        | LeRobot format (v0.1.0)               | Required for VLA use cases                |
| Connection model      | Explicit connection strings only (v1) | No discovery layer complexity             |

## Guardrails

- No heuristic robot detection (e.g., based on action length)
- No hardware SDKs in core library (optional extras only)
- No user-facing adapter configuration required
- No offline buffering / async upload queue in v1
- No application dependency for edge deployment
- No core logic in physical‑ai‑framework (thin wrappers only)

---

## Package Naming

| Package           | Name                    | Status                               |
| ----------------- | ----------------------- | ------------------------------------ |
| Generic Inference | `inferencekit`          | Proposed replacement for `model_api` |
| Robotics Stack    | `getiaction`            | Library + application                |
| Deployment Shell  | `physical‑ai‑framework` | Codename for deployment repo         |

_Names are subject to marketing/branding review._

---

## Future Considerations

- Multi-arm + shared camera semantics (composition vs single robot abstraction)
- Camera device identification (`device_key` standardization)
- Long-term discovery strategy
- Timing guarantees for control loop rate
- Unified profile schema for common workflows

---

## Component Documentation

### Library Components

| Component        | Document                                            | Description                                                 |
| ---------------- | --------------------------------------------------- | ----------------------------------------------------------- |
| Robot Interface  | [Robot Interface](./library/robot-interface.md)     | Robot ABC, leader/follower wrappers, SDK integration        |
| Camera Interface | [Camera Interface](./library/camera-interface.md)   | Camera ABC, invisible sharing, callbacks, capability mixins |
| Teleoperation    | [Teleoperation API](./library/teleoperation.md)     | Leader/follower semantics, session lifecycle, safety        |
| Data Collection  | [Data Collection API](./library/data-collection.md) | DatasetWriter, episode management, HF Hub upload            |

### Deployment Stack

| Component           | Document                                             | Description                             |
| ------------------- | ---------------------------------------------------- | --------------------------------------- |
| Inference Engine    | [inferencekit](./deployment/inferencekit.md)         | Domain-agnostic inference framework     |
| Deployment Shell    | [Deployment Shell](./deployment/deployment-shell.md) | physical‑ai‑framework CLI and packaging |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md)       | PolicyPackage plugin, runner mapping    |

---

_Last Updated: 2026-02-06_
