---
marp: true
theme: default
paginate: true
backgroundColor: #fff
title: "Proposal: Physical-AI Inference Engine"
---

# Proposal: Physical‑AI Inference Engine

## A unified deployment layer for physical‑AI policies

---

## The Problem Today

Every team re‑implements the same deployment plumbing:

- Camera capture + observation construction
- Safety checks + action clamping
- Episode loops + reset logic
- Device lifecycle + cleanup
- Backend selection + model loading

**Result:** Duplicated effort, inconsistent safety, fragile integrations.

---

## What We Propose

A **shared deployment layer** — called **physical‑ai‑framework** — that owns the common plumbing.

Teams write only the parts unique to their model:

- A **Runner** (how to execute the forward pass)
- A **Preprocessor** / **Postprocessor** (how to shape inputs/outputs)

The framework handles everything else.

---

## Where the Code Lives Today

There is no physical‑ai‑framework package yet. There is no inferencekit package yet.

**Today:** All inference code lives inside **getiaction**. Model loading, backend adapters, metadata parsing, runners — all in one repo.

**What we propose:** Extract and layer.

```
getiaction (monolith today)
    │
    ├── inference core ──────► extract as "inferencekit" (domain-agnostic base)
    ├── physical-AI logic ───► new package "physical-ai-framework" (robotics layer)
    └── training, export ────► stays in getiaction
```

This is a **design proposal**, not a status report. Nothing below exists as shipped code.

---

## Target Experience: Python API

```python
from physical_ai import InferenceModel

# GetiAction user
policy = InferenceModel("hf://getiaction/act_policy")

# LeRobot user
policy = InferenceModel("hf://lerobot/pi0")

# Custom local model
policy = InferenceModel("./exports/dreamzero")
```

**Three personas. One API. Same behavior.**

---

## Target Experience: CLI

```bash
# GetiAction model
phyai run --model hf://getiaction/act_policy --robot robot.yaml

# LeRobot model
phyai run --model hf://lerobot/pi0 --robot robot.yaml

# Custom model
phyai run --model ./exports/dreamzero --robot robot.yaml
```

**Three models. One CLI. Same flags.**

---

## Today vs Proposed

**Today — each team owns the full stack:**

```
getiaction: camera → obs → preprocess → infer → postprocess → robot
lerobot:    camera → obs → preprocess → infer → postprocess → robot
custom:     camera → obs → preprocess → infer → postprocess → robot
```

Duplicated. Inconsistent. Each team re‑solves safety, device lifecycle, observation buffering.

**Proposed — shared framework, team-owned plugins:**

```
framework:  camera → obs ──────────────────────────── → robot
plugin:                  └→ preprocess → infer → post ┘
```

Teams own only the unique part.

---

## Proposed Scope for physical‑ai‑framework

| Capability                | Framework would provide                           | Teams still own               |
| ------------------------- | ------------------------------------------------- | ----------------------------- |
| **Observation pipeline**  | Camera → observation dict, buffering, timestamps  | Custom observation transforms |
| **Safety runtime**        | Action clamp, velocity limits, e‑stop             | Domain‑specific constraints   |
| **Episode orchestration** | Run N episodes, reset, log                        | Termination conditions        |
| **Device management**     | Robot/camera lifecycle, cleanup                   | SDK driver implementations    |
| **Validation CLI**        | `phyai validate` — metadata, class_paths, dry‑run | Model‑specific validation     |

This is more than a plugin loader. It's the **shared runtime** that makes deployment consistent.

---

## Proposed Architecture: Three Packages

```
getiaction → physical‑ai‑framework → inferencekit
                    │
                    ├── physical_ai.camera  (clean subpackage)
                    ├── physical_ai.robot   (clean subpackage)
                    └── physical_ai.engine  (plugins, CLI, safety)
```

| Package                   | What it would contain                                     | Depends on              |
| ------------------------- | --------------------------------------------------------- | ----------------------- |
| **inferencekit**          | Extracted from getiaction: adapters, runners, metadata IO | Nothing domain‑specific |
| **physical‑ai‑framework** | New: observation, safety, episodes, devices, CLI          | inferencekit            |
| **getiaction**            | Remains: training, export, domain pre/post                | physical‑ai‑framework   |

---

## What Is inferencekit? (Extraction, Not Greenfield)

**inferencekit** is not a new idea — it's the inference core that **already exists inside getiaction**, extracted into its own package so other domains can use it.

What gets extracted:

- `InferenceModel` — load model, run inference
- `RuntimeAdapter` — one forward pass on a specific backend (OpenVINO, ONNX, TensorRT)
- `InferenceRunner` — execution patterns (single-pass, iterative, tiled)
- `Preprocessor` / `Postprocessor` ABCs — transform inputs/outputs
- `Callbacks` — cross-cutting hooks (timing, logging)
- Metadata loading — YAML/JSON → class_path + init_args

**inferencekit would know nothing about vision, robotics, or any domain.** Domain logic lives in layers above it.

---

## inferencekit: Proposed Layered Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Domain Layers                          │
│                                                          │
│  model_api       physical‑ai‑framework     custom‑xyz   │
│  (vision)        (physical‑AI)             (your domain) │
│                                                          │
│           └──────────┼──────────┘                        │
│                      │                                   │
│               depends on                                 │
│                      ▼                                   │
│  ┌──────────────────────────────────────────────────┐    │
│  │                 inferencekit                      │    │
│  │         (extracted from getiaction)               │    │
│  │                                                   │    │
│  │  InferenceModel │ RuntimeAdapter │ InferenceRunner│    │
│  │  Callbacks      │ Pre/Post ABCs  │ Plugin Registry│    │
│  │  OpenVINO, ONNX, TensorRT, Torch backends        │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

Multiple domain layers can share the same inference base — vision, robotics, or anything else.

---

## Why One Direction Matters

```
getiaction ──hard depends──► physical‑ai‑framework ──hard depends──► inferencekit
                                      │
                        loads at runtime (class_path)
                                      │
                              getiaction plugins
```

- **Hard dependency:** `import` at module level. Declared in `pyproject.toml`.
- **Runtime loading:** `class_path` string in metadata. No import. No dependency edge.

**getiaction would always need physical‑ai‑framework.**
**physical‑ai‑framework would never need getiaction.**

No circular dependencies. One direction only.

---

## Proposed Interface Ownership

We propose physical‑ai‑framework owns camera/robot interfaces as **clean subpackages**:

```
physical_ai/
├── camera/    ← CameraBase ABC, zero imports from rest of physical_ai
├── robot/     ← RobotBase ABC, zero imports from rest of physical_ai
└── engine/    ← plugins, CLI, safety, episodes
```

**Why subpackages, not separate repos?**

- Versioned together with the framework that uses them
- Import linting enforces zero cross‑imports
- 3 repos (inferencekit + framework + getiaction), not 5

---

## How Plugins Would Work: Metadata

```yaml
# exports/act_policy/metadata.yaml
backend: openvino
runner:
  class_path: getiaction.runners.ActionChunkingRunner
  init_args:
    chunk_size: 100

preprocessors:
  - class_path: getiaction.pre.ObservationNormalizer
    init_args:
      mean: [0.485, 0.456, 0.406]

postprocessors:
  - class_path: getiaction.post.ActionClamp
    init_args:
      limits: [-1.0, 1.0]
```

**`class_path` = fully qualified Python class. Installed via pip. No entry points required.**

---

## How Plugins Would Work: Discovery Flow

```
model path / URI
       │
       ▼
metadata detection (metadata.yaml → metadata.yml → metadata.json → manifest.json)
       │
       ▼
class_path resolution (importlib)
       │
       ▼
Runner + Preprocessors + Postprocessors instantiated
       │
       ▼
InferenceModel ready
```

**First metadata file wins. class_path must be importable (pip installed).**

---

## How Plugins Would Work: Runtime Pipeline

```
observation (from framework)
       │
       ▼
preprocessors[0] → ... → preprocessors[N]
       │
       ▼
runner.run(adapter, inputs)
       │         │
       │    adapter.predict(inputs)  ← one forward pass (OV / ONNX / TRT)
       │         │
       │         ▼
       │    raw outputs
       ▼
postprocessors[0] → ... → postprocessors[N]
       │
       ▼
action (to framework → robot)
```

---

## What's Inside an Export Directory

```
exports/act_policy/
├── metadata.yaml          # plugin wiring (class_path + init_args)
├── model.onnx             # ONNX backend
├── model.xml              # OpenVINO IR
├── model.bin              # OpenVINO weights
└── artifacts/             # optional extra files
```

The runtime would pick backend via `backend` / `device` fields in metadata (or CLI flags).

---

## Proposed Backend Support

| Hardware              | Backend      | Adapter             |
| --------------------- | ------------ | ------------------- |
| Intel CPU/GPU         | OpenVINO     | `OpenVINOAdapter`   |
| Cross‑platform (CUDA) | ONNX Runtime | `ONNXAdapter`       |
| NVIDIA GPU            | TensorRT     | `TensorRTAdapter`   |
| Edge / mobile         | ExecuTorch   | `ExecuTorchAdapter` |

```bash
# Target CLI experience — override backend at runtime
phyai run --model ./exports/act_policy --backend openvino --device CPU
```

---

## Custom Model: Zero Code (Target: 5 Minutes)

**Standard ONNX model — no plugin needed:**

```yaml
# exports/my_model/metadata.yaml
backend: onnx
runner:
  class_path: inferencekit.runners.SinglePassRunner
preprocessors: []
postprocessors: []
```

```python
model = InferenceModel("./exports/my_model")
outputs = model(inputs)
```

**Target:** Works immediately. No custom code. No pip install beyond the framework.

---

## Custom Model: Custom Logic (Target: 15 Minutes)

**Non‑standard execution pattern — write a plugin:**

1. Create package with Runner / Preprocessor / Postprocessor
2. `pip install -e ./my_plugin`
3. Point `class_path` in `metadata.yaml` at your classes
4. Run: `phyai run --model ./exports/my_model --robot robot.yaml`

**Plugin ownership:** Your team writes it. The framework loads it. No PR to our repos needed.

---

## Proposed: Validate Without Hardware

```bash
phyai validate ./exports/my_model
```

```
✓ metadata.yaml found
✓ class_path getiaction.runners.ActionChunkingRunner resolves
✓ class_path getiaction.pre.ObservationNormalizer resolves
✓ dry-run inference passes (random input)
✓ output shape matches expected action dimensions
```

**Catch errors before touching a robot.**

---

## Where Does My Logic Go?

| I need to...                              | Implement         | Base class          |
| ----------------------------------------- | ----------------- | ------------------- |
| Custom forward pass (chunking, iterative) | **Runner**        | `InferenceRunner`   |
| Transform inputs before inference         | **Preprocessor**  | `Preprocessor`      |
| Transform outputs after inference         | **Postprocessor** | `Postprocessor`     |
| Cross‑cutting (timing, logging, safety)   | **Callback**      | `InferenceCallback` |

**Rule of thumb:** If it touches the model → Runner. If it shapes data → Pre/Post. If it observes → Callback.

---

## When Plugins Aren't Enough: Subclass InferenceModel

Plugins would cover most cases. **Subclass when orchestration itself must change:**

| Need                                                     | Why subclass                                 |
| -------------------------------------------------------- | -------------------------------------------- |
| Multi-model pipeline (model A feeds model B)             | Orchestration between steps changes          |
| Custom lifecycle (`warm_up()`, `reset()`, `calibrate()`) | New methods the base class doesn't have      |
| Stateful inference (episode buffer, history window)      | State lives outside the single-call pipeline |

```python
from physical_ai import InferenceModel

class PerceptionPolicyModel(InferenceModel):
    """Two-stage: perception feeds policy."""
    def __init__(self, policy_path, perception_path):
        super().__init__(policy_path)
        self.perception = InferenceModel(perception_path)

    def __call__(self, raw_observation):
        features = self.perception(raw_observation)
        return super().__call__(features)
```

**Rule:** Always call `super()`. Never bypass the plugin pipeline.

---

## Packaging Independence

| Level          | What you do                         | Upstream needed? |
| -------------- | ----------------------------------- | ---------------- |
| **Local**      | `pip install -e ./my_plugin`        | No               |
| **Published**  | `pip install my-plugin` (PyPI)      | No               |
| **Upstreamed** | Add entry points for auto‑discovery | Optional PR      |

**Day 1:** Editable install, test locally.
**Day 30:** Publish to PyPI, other teams use it.
**Day 60 (optional):** Upstream entry points for auto‑discovery.

`class_path` in metadata **always works** — entry points are convenience, not requirement.

---

## Proposed Adoption Timeline

| Milestone   | What happens                                                                           |
| ----------- | -------------------------------------------------------------------------------------- |
| **Phase 1** | Extract inferencekit from getiaction. Separate package, same functionality.            |
| **Phase 2** | Build physical‑ai‑framework: observation pipeline, safety, episode orchestration, CLI. |
| **Phase 3** | Migrate getiaction deployment to use the framework. getiaction becomes a plugin.       |
| **Phase 4** | Onboard LeRobot and custom models as plugins.                                          |

---

## Current State vs Proposed Target

| Area                        | Today (in getiaction)  | Proposed (in framework)                        |
| --------------------------- | ---------------------- | ---------------------------------------------- |
| **Inference core**          | Embedded in getiaction | Extracted as inferencekit (shared)             |
| **Observation pipeline**    | Each team builds own   | Framework provides camera → obs dict           |
| **Safety runtime**          | Ad‑hoc per deployment  | Framework enforces clamp / velocity / e‑stop   |
| **Episode orchestration**   | Manual loops           | Framework runs N episodes with reset + logging |
| **Device management**       | Scattered cleanup      | Framework owns lifecycle                       |
| **Validation**              | None                   | `phyai validate` catches errors pre‑deploy     |
| **Multi-framework support** | getiaction only        | getiaction, LeRobot, custom — same API         |

---

## Proposed Installation

```bash
# Framework only (for custom models)
pip install physical-ai-framework

# With GetiAction plugin
pip install physical-ai-framework[getiaction]

# With LeRobot plugin
pip install physical-ai-framework[lerobot]
```

`physical-ai-framework[getiaction]` would **not** be a circular dependency — it's a pip convenience extra that pulls in the getiaction plugin package.

---

## Key Decisions (Requesting Alignment)

| Decision               | Proposed choice                              | Why                                                     |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------- |
| Extract inference core | inferencekit as separate package             | Enables vision + robotics + custom domains on same base |
| Interface ownership    | physical‑ai‑framework owns camera/robot ABCs | One direction. No circular deps.                        |
| Subpackages vs repos   | Subpackages with import linting              | Versioned together. 3 repos not 5.                      |
| Plugin mechanism       | `class_path` in metadata                     | Works without entry points. Day‑1 ready.                |
| Framework scope        | Observation + safety + episodes + devices    | Not just a plugin loader. Real value.                   |
| Dependency direction   | getiaction → framework → inferencekit        | Always one‑way. Framework never imports getiaction.     |

---

## Summary

1. **The problem is real** — teams duplicate deployment plumbing today
2. **We propose two new packages** — inferencekit (extracted from getiaction) and physical‑ai‑framework (new)
3. **One API, many plugins** — GetiAction, LeRobot, and custom models would deploy the same way
4. **Framework owns the hard parts** — observation, safety, episodes, devices, validation
5. **No circular dependencies** — one‑directional: getiaction → framework → inferencekit
6. **Incremental path** — extract first, build framework second, migrate third

---

# Q&A

Design docs: `docs/design/deployment/`
