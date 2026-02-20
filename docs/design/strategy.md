# Strategy: Geti Action Architecture

## Executive Summary

Geti Action is an end-to-end platform for robot AI development: data collection, training, and deployment. It consists of two packages — a **library** and an **application** — with clear ownership boundaries.

**Two core architectural decisions:**

1. **Library-first**: The library owns every component needed for end-to-end robot AI (robots, cameras, teleop, data collection, policies, inference, training, export). The application is purely UI and orchestration — glue on top.

2. **Layered deployment stack**: For inference/deployment, we propose a clean layering — **physical‑ai‑framework** (universal physical‑AI inference engine with a domain‑agnostic inference core as an internal modular layer, unified `manifest.json` format, and built‑in runners) → **external plugins** (only for exotic execution patterns). Vision remains a separate domain layer (model_api) that can share the inference core.

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
│  - Built-in format loaders and runners                      │
│  - Observation pipeline, safety runtime, episode orchestration│
│  - Camera/robot interfaces (clean subpackages)              │
│  - CLI (run/serve/export/validate)                          │
│                                                             │
│  pip install physical-ai-framework                          │
│  phyai run --config deploy.yaml                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │      physical_ai.inference (inference core)          │    │
│  │      Domain-agnostic modular layer:                  │    │
│  │      RuntimeAdapter, backend abstraction (OpenVINO,  │    │
│  │      ONNX, TensorRT, Torch), metadata IO,           │    │
│  │      base InferenceModel.                            │    │
│  │      No robotics, no vision, no domain logic.        │    │
│  │      Can be silently extracted as a separate package  │    │
│  │      later if other domains need it standalone.      │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Domain Layers (build on top)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     Domain Layers                           │
│                                                             │
│  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │     model_api          │  │   physical-ai plugins     │  │
│  │     (vision)           │  │   (external, exotic only) │  │
│  │                        │  │  exotic pre/post, runners │  │
│  │  YOLO, SAM, Anomaly,   │  │  for patterns not covered │  │
│  │  image preprocessing,  │  │  by built‑in components   │  │
│  │  NMS, result types     │  │                           │  │
│  └────────────────────────┘  └───────────────────────────┘  │
│                                                             │
│  model_api can share the inference core                     │
│  (physical_ai.inference) for backend execution.             │
└─────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer                     | Owns                                                                                                                                                                                             | Does NOT own                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| **physical_ai.inference** | Runtime adapters, backend abstraction, manifest IO, base InferenceModel (domain‑agnostic modular layer inside physical‑ai‑framework)                                                             | Vision, robotics, domain logic |
| **model_api**             | Vision preprocessing, task wrappers (YOLO, SAM), result types, image-specific pipelines                                                                                                          | Backend execution, robotics    |
| **physical‑ai‑framework** | Physical‑AI orchestration, unified APIs, built‑in format loaders, built‑in runners, observation pipeline, safety runtime, episode orchestration, device management, camera/robot interfaces, CLI | Training, vision models        |
| **external plugins**      | Exotic pre/post, runners for patterns not covered by built‑in components (user's own package)                                                                                                    | Backend execution              |

### physical‑ai‑framework as a Universal Physical‑AI Engine

physical‑ai‑framework is not a thin shell. It is the **universal inference engine for physical‑AI** — it owns every domain concern common to all physical‑AI deployments so that teams only supply their model-specific logic:

- **Unified API surface** (`InferenceModel`) for all physical‑AI policies
- **Unified manifest format** (`manifest.json`) for all model sources — getiaction, LeRobot, and custom
- **Built‑in runners** (SinglePassRunner, IterativeRunner, ActionChunkingRunner) for common execution patterns
- **Observation pipeline** — camera → observation dict, timestamp alignment, buffering
- **Safety runtime** — action clamping, velocity limits, workspace bounds, emergency stop (first-class engine layer, not a callback)
- **Episode orchestration** — run N episodes, reset between episodes, log results
- **Device management** — robot/camera connection lifecycle, cleanup on error
- **Validation CLI** — `phyai validate` to verify metadata, resolve class_paths, dry-run pipeline without hardware
- **CLI** for edge and server inference (`phyai run`, `phyai serve`, `phyai export`)

Most models work with the unified manifest format and built‑in runners — zero external dependencies. Exotic execution patterns use external plugins (user's own package). Backend execution lives in the inference core (`physical_ai.inference`), a domain‑agnostic modular layer inside the framework. Camera and robot interfaces live as clean subpackages inside physical‑ai‑framework (`physical_ai.camera`, `physical_ai.robot`).

### Why This Layering?

1. **The inference core** (`physical_ai.inference`) is reusable beyond robotics (vision, NLP, anomaly detection). Keeping it domain-agnostic as a clean modular layer maximizes reuse and allows silent extraction as a standalone package later.
2. **physical‑ai‑framework** centralizes physical‑AI orchestration without re-implementing backends or training code.
3. **Built‑in format loaders and runners** allow rapid support for new models without any external dependencies. External plugins handle exotic patterns.

### Key Rule

Dependencies flow **upward only**:

```
getiaction → physical‑ai‑framework
```

getiaction depends on physical‑ai‑framework for camera/robot interfaces and the engine runtime. The inference core (`physical_ai.inference`) lives inside physical‑ai‑framework as a domain‑agnostic modular layer. physical‑ai‑framework loads models at runtime via the unified `manifest.json` format and `class_path` — never imports getiaction at install time. External plugins are optional and can live in their own repos. The inference core stays domain‑agnostic.

### Why getiaction Inference Stays in getiaction

getiaction remains the source of truth for its own policies. Its exported models use the unified `manifest.json` format which the framework's manifest loader reads natively. Built‑in runners (like ActionChunkingRunner) handle getiaction execution patterns. No getiaction import is needed at deployment time.

### Robot/Camera APIs: Interface Ownership

Both **getiaction** and **physical‑ai‑framework** need access to robot + camera APIs. Two options exist. We recommend Option 1.

#### Option 1 (Recommended): physical‑ai‑framework owns the interfaces

```
getiaction → physical‑ai‑framework
                   │
                   ├── physical_ai.camera     (clean subpackage)
                   ├── physical_ai.robot      (clean subpackage)
                   ├── physical_ai.inference  (domain-agnostic modular layer)
                   └── physical_ai.engine     (format loaders, runners, CLI, safety)
```

**Why:**

- **No circular dependency.** getiaction depends on physical‑ai‑framework. physical‑ai‑framework loads models at runtime via the unified manifest format and `class_path` — never imports getiaction at install time. One-directional dependency.
- **Fewer repos (2 instead of 5+).** Only physical‑ai‑framework and getiaction. Less coordination overhead, simpler CI, fewer version matrices.
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

**Cost:** 5+ repos instead of 2. Every release requires coordinating versions across camera-api, robot-api, physical-ai-framework, and getiaction.

**Verdict:** Start with Option 1. Split when a concrete consumer forces it.

### Extension Contract (Built‑in + External)

The framework ships a unified `manifest.json` format and built‑in runners for common patterns. For exotic models, external plugins provide:

- **Preprocessors**: observation → model inputs
- **Runners**: policy‑specific execution patterns (beyond SinglePass, Iterative, ActionChunking)
- **Postprocessors**: model outputs → actions
- **Optional wrapper**: policy‑specific API (`select_action`, etc.)

External plugins can be local (editable install), internal, or published. Most models need zero external plugins.

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

| Decision              | Choice                                                        | Rationale                                        |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------ |
| Core logic location   | getiaction library                                            | Single source of truth, no version skew          |
| Application role      | UI + orchestration only                                       | Library-first; app adds no core logic            |
| Deployment packaging  | Universal engine + unified manifest format + built‑in runners | Supports any model framework, zero external deps |
| Robot/camera API home | Library, not application                                      | Edge deployment without web server               |
| Installation model    | Optional extras per SDK                                       | Lightweight core, no SDK pollution               |
| Async strategy        | Async core + sync wrapper                                     | App needs async, CLI needs sync                  |
| CLI framework         | LightningCLI / jsonargparse                                   | Align with existing getiaction patterns          |
| Dataset format        | LeRobot format (v0.1.0)                                       | Required for VLA use cases                       |
| Connection model      | Explicit connection strings only (v1)                         | No discovery layer complexity                    |

## Guardrails

- No heuristic robot detection (e.g., based on action length)
- No hardware SDKs in core library (optional extras only)
- No user-facing adapter configuration required
- No offline buffering / async upload queue in v1
- No application dependency for edge deployment
- No training code in physical‑ai‑framework

---

## Package Naming

| Package             | PyPI name          | Status                                                                                                      |
| ------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------- |
| Runtime / Inference | `physicalai`       | Lightweight runtime — inference, camera, robot, export, benchmark runner. Published from `physical-ai` repo |
| Training SDK        | `physicalai-train` | Training, policies, data, eval, gyms, benchmark presets. Published from `physical-ai-studio` repo           |
| Vision Layer        | `model_api`        | Vision inference layer; can share framework's inference core                                                |

Both `physicalai` and `physicalai-train` share the `physicalai` namespace via PEP 420 implicit namespace packaging. See [Packaging Strategy](./deployment/physical-ai-two-repo-options.md) for the full rollout plan.

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

| Component           | Document                                                           | Description                                                    |
| ------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| Packaging Strategy  | [Packaging Strategy](./deployment/physical-ai-two-repo-options.md) | Two-repo, two-distribution packaging with PEP 420 namespace    |
| Inference Core      | [Inference Core](./deployment/inferencekit.md)                     | Domain-agnostic inference layer design (internal to framework) |
| Deployment Engine   | [Deployment Engine](./deployment/deployment-shell.md)              | physical‑ai‑framework universal engine, CLI, format loaders    |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md)                     | LeRobot integration via unified manifest.json format           |

---

_Last Updated: 2026-02-20_
