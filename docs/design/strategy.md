# Strategy: Geti Action Architecture

## Executive Summary

Geti Action is an end-to-end platform for robot AI development: data collection, training, and deployment. It consists of two packages — a **library** and an **application** — with clear ownership boundaries.

**Two core architectural decisions:**

1. **Library-first**: The library owns every component needed for end-to-end robot AI (robots, cameras, teleop, data collection, policies, inference, training, export). The application is purely UI and orchestration — glue on top.

2. **Layered deployment stack**: For inference/deployment, we propose a clean layering — **inferencekit** (base execution engine) → **physical‑ai‑framework** (universal physical‑AI inference engine) → **plugins** (getiaction, LeRobot, custom frameworks). Vision remains a separate domain layer (model_api) on top of inferencekit.

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

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         physical‑ai‑framework (universal engine)            │
│                                                             │
│  Physical‑AI inference engine:                              │
│  - Unified API (InferenceModel)                             │
│  - Policy plugin registry (getiaction, LeRobot, custom)     │
│  - Observation pipeline, safety runtime, episode orchestration│
│  - Camera/robot interfaces (clean subpackages)              │
│  - CLI (run/serve/export/validate)                          │
│                                                             │
│  pip install physical-ai-framework                          │
│  phyai run --config deploy.yaml                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ depends on
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    inferencekit (base)                       │
│                                                             │
│  Domain-agnostic execution engine:                          │
│  RuntimeAdapter, backend abstraction (OpenVINO, ONNX,       │
│  TensorRT, Torch), metadata IO, base InferenceModel.        │
│  No robotics, no vision, no domain logic.                   │
│                                                             │
│  from inferencekit import InferenceModel                    │
│  model = InferenceModel("./exports/my_model")               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Domain Layers                           │
│                                                             │
│  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │     model_api          │  │   physical-ai plugins     │  │
│  │     (vision)           │  │   (getiaction, LeRobot,   │  │
│  │                        │  │    custom frameworks)     │  │
│  │  YOLO, SAM, Anomaly,   │  │  policy-specific pre/post,│  │
│  │  image preprocessing,  │  │  runners, wrappers        │  │
│  │  NMS, result types     │  │                           │  │
│  └───────────┬────────────┘  └─────────────┬─────────────┘  │
│              │                             │                │
│              └──────────┬──────────────────┘                │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │ depends on
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    inferencekit (base)                       │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer                     | Owns                                                                                                                                                                          | Does NOT own                   |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **inferencekit**          | Runtime adapters, backend abstraction, metadata IO, base InferenceModel                                                                                                       | Vision, robotics, domain logic |
| **model_api**             | Vision preprocessing, task wrappers (YOLO, SAM), result types, image-specific pipelines                                                                                       | Backend execution, robotics    |
| **physical‑ai‑framework** | Physical‑AI orchestration, unified APIs, policy plugin registry, observation pipeline, safety runtime, episode orchestration, device management, camera/robot interfaces, CLI | Training, vision models        |
| **physical‑ai plugins**   | Policy‑specific pre/post, runners, wrappers (getiaction, LeRobot, custom)                                                                                                     | Backend execution              |

### physical‑ai‑framework as a Universal Physical‑AI Engine

physical‑ai‑framework is not a thin shell. It is the **universal inference engine for physical‑AI** — it owns every domain concern common to all physical‑AI deployments so that teams only supply their model-specific logic:

- **Unified API surface** (`InferenceModel`) for all physical‑AI policies
- **Policy plugin registry** to load getiaction, LeRobot, and custom frameworks
- **Observation pipeline** — camera → observation dict, timestamp alignment, buffering
- **Safety runtime** — action clamping, velocity limits, workspace bounds, emergency stop (first-class engine layer, not a callback)
- **Episode orchestration** — run N episodes, reset between episodes, log results
- **Device management** — robot/camera connection lifecycle, cleanup on error
- **Validation CLI** — `phyai validate` to verify metadata, resolve class_paths, dry-run pipeline without hardware
- **CLI** for edge and server inference (`phyai run`, `phyai serve`, `phyai export`)

All policy-specific logic lives in plugins. All backend execution lives in inferencekit. Camera and robot interfaces live as clean subpackages inside physical‑ai‑framework (`physical_ai.camera`, `physical_ai.robot`).

### Why This Layering?

1. **inferencekit** is reusable beyond robotics (vision, NLP, anomaly detection). Keeping it domain-agnostic maximizes reuse.
2. **physical‑ai‑framework** centralizes physical‑AI orchestration without re-implementing backends or training code.
3. **Plugins** allow rapid support for new policies without changing the engine.

### Key Rule

Dependencies flow **upward only**:

```
getiaction → physical‑ai‑framework → inferencekit
model_api                          → inferencekit
```

getiaction depends on physical‑ai‑framework for camera/robot interfaces and the engine runtime. physical‑ai‑framework depends on inferencekit for backend execution. physical‑ai‑framework loads getiaction plugins at runtime via `class_path` / entry points — never imports getiaction at install time. Plugins are optional and can live in their own repos. inferencekit stays domain‑agnostic.

### Why getiaction Inference Stays in getiaction

getiaction remains the source of truth for its own policies. Its inference pipeline lives in getiaction and is exposed to physical‑ai‑framework as a plugin. This avoids duplication and keeps training/inference aligned.

### Robot/Camera APIs: Interface Ownership

Both **getiaction** and **physical‑ai‑framework** need access to robot + camera APIs. Two options exist. We recommend Option 1.

#### Option 1 (Recommended): physical‑ai‑framework owns the interfaces

```
getiaction → physical‑ai‑framework → inferencekit
                   │
                   ├── physical_ai.camera  (clean subpackage)
                   ├── physical_ai.robot   (clean subpackage)
                   └── physical_ai.engine  (plugin system, CLI, safety)
```

**Why:**

- **No circular dependency.** getiaction depends on physical‑ai‑framework. physical‑ai‑framework loads getiaction plugins at runtime via `class_path` / entry points — never imports getiaction at install time. One-directional dependency.
- **Fewer repos (3 instead of 5+).** Only inferencekit, physical‑ai‑framework, and getiaction. Less coordination overhead, simpler CI, fewer version matrices.
- **One package for all hardware interfaces.** Teams install physical‑ai‑framework and get cameras, robots, inference, CLI, safety — everything needed for deployment.
- **Future split is cheap.** Camera/robot interfaces live in clean subpackages with no cross-imports. If a vision-only consumer needs camera-api standalone, extract it then. Merging repos later is much harder than splitting.

**Condition:** Camera/robot subpackages must have **zero imports** from the rest of physical‑ai‑framework. Enforced by import linting. This makes future extraction trivial.

**Trade-off:** `pip install getiaction` pulls physical‑ai‑framework as a dependency. Acceptable because getiaction needs hardware interfaces for training (teleoperation, data collection) anyway.

#### Option 2 (Alternative): Shared interfaces in separate packages

```
camera‑api   robot‑api   inferencekit (base)
    ▲            ▲              ▲
    │            │              │
    ├────────────┼──────────────┤
    │            │              │
getiaction   physical‑ai‑framework
 (training)         (engine)
```

**When to prefer Option 2:**

- You have **concrete vision-only consumers** that need camera-api without physical-ai-framework.
- You need getiaction installable **without** physical-ai-framework (e.g., CI environments that only run training).
- Multiple teams independently maintain camera and robot interfaces with separate release cadences.

**Cost:** 5+ repos instead of 3. Every release requires coordinating versions across camera-api, robot-api, inferencekit, physical-ai-framework, and getiaction.

**Verdict:** Start with Option 1. Split when a concrete consumer forces it.

### Plugin Contract (Summary)

Each physical‑AI plugin provides:

- **Preprocessors**: observation → model inputs
- **Runners**: policy‑specific execution patterns (chunking, diffusion)
- **Postprocessors**: model outputs → actions
- **Optional wrapper**: policy‑specific API (`select_action`, etc.)

Plugins can be local (editable install), internal, or published.

---

## User-Facing Experience

### Inference (Library API)

**Unified API (raw inputs/outputs):**

```python
from physical_ai import InferenceModel

model = InferenceModel("hf://getiaction/act_policy")
outputs = model(model_inputs)
action = outputs["action"]
```

**Policy API (observation → action):**

```python
from physical_ai import InferenceModel

policy = InferenceModel("hf://getiaction/act_policy")
action = policy.select_action(observation)
```

### CLI — Deployment (physical‑ai‑framework)

```bash
phyai run --model ./exports/act_policy --robot robot.yaml --episodes 10
phyai serve --model ./exports/act_policy --robot robot.yaml
```

---

## Key Decisions

| Decision              | Choice                                 | Rationale                               |
| --------------------- | -------------------------------------- | --------------------------------------- |
| Core logic location   | getiaction library                     | Single source of truth, no version skew |
| Application role      | UI + orchestration only                | Library-first; app adds no core logic   |
| Deployment packaging  | Universal physical‑AI engine + plugins | Supports any policy framework           |
| Robot/camera API home | Library, not application               | Edge deployment without web server      |
| Installation model    | Optional extras per SDK                | Lightweight core, no SDK pollution      |
| Async strategy        | Async core + sync wrapper              | App needs async, CLI needs sync         |
| CLI framework         | LightningCLI / jsonargparse            | Align with existing getiaction patterns |
| Dataset format        | LeRobot format (v0.1.0)                | Required for VLA use cases              |
| Connection model      | Explicit connection strings only (v1)  | No discovery layer complexity           |

## Guardrails

- No heuristic robot detection (e.g., based on action length)
- No hardware SDKs in core library (optional extras only)
- No user-facing adapter configuration required
- No offline buffering / async upload queue in v1
- No application dependency for edge deployment
- No training code in physical‑ai‑framework

---

## Package Naming

| Package           | Name                    | Status                                                  |
| ----------------- | ----------------------- | ------------------------------------------------------- |
| Base Inference    | `inferencekit`          | Base inference framework; domain layers build on top    |
| Vision Layer      | `model_api`             | Vision inference layer on top of inferencekit           |
| Robotics Stack    | `getiaction`            | Robotics inference layer + library + application        |
| Deployment Engine | `physical‑ai‑framework` | Universal physical‑AI inference engine + plugin runtime |

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

| Component           | Document                                              | Description                                                  |
| ------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Inference Engine    | [inferencekit](./deployment/inferencekit.md)          | Base inference framework with plugin system                  |
| Deployment Engine   | [Deployment Engine](./deployment/deployment-shell.md) | physical‑ai‑framework universal engine, CLI, plugin registry |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md)        | PolicyPackage plugin, runner mapping                         |

---

_Last Updated: 2026-02-11_
