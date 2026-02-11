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

For most exported models, teams write **nothing** — built‑in runners and preprocessors handle it.

For exotic execution patterns, teams write:

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

**Proposed — shared framework, built‑in runners:**

```
framework:  camera → obs ──────────────────────────── → robot
built-in:                └→ preprocess → infer → post ┘
```

Built‑in runners handle common patterns. Teams own nothing unless their model is exotic.

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
                     └── physical_ai.engine  (format loaders, built‑in runners, CLI, safety)
```

| Package                   | What it would contain                                                              | Depends on              |
| ------------------------- | ---------------------------------------------------------------------------------- | ----------------------- |
| **inferencekit**          | Extracted from getiaction: adapters, runners, metadata IO                          | Nothing domain‑specific |
| **physical‑ai‑framework** | New: format loaders, built‑in runners, observation, safety, episodes, devices, CLI | inferencekit            |
| **getiaction**            | Remains: training, export, domain pre/post (as external plugin if exotic)          | physical‑ai‑framework   |

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
                         external plugins (if exotic)
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

## Two Tiers: Format Loaders vs External Plugins

| Tier                 | What                                                               | Ships with framework?   | Extra dependencies? |
| -------------------- | ------------------------------------------------------------------ | ----------------------- | ------------------- |
| **Format loaders**   | Parse `metadata.yaml` or `manifest.json` → same internal structure | Yes, built‑in           | None                |
| **Built‑in runners** | `SinglePassRunner`, `IterativeRunner`, `ActionChunkingRunner`      | Yes, built‑in           | None                |
| **External plugins** | Custom Runner / Pre / Post for exotic models                       | No — user's own package | User's choice       |

**Most exported models need zero external packages.** The framework understands common metadata formats and ships runners for common execution patterns. External plugins are only for truly exotic cases.

---

## Why No LeRobot Plugin Is Needed

LeRobot exports a `manifest.json` with `policy.kind: "single_shot"` or `"iterative"`.

The framework reads that manifest (pure JSON, no lerobot import) and maps it to **our built‑in runners**:

| LeRobot `policy.kind` | Built‑in runner    | lerobot dependency? |
| --------------------- | ------------------ | ------------------- |
| `single_shot`         | `SinglePassRunner` | No                  |
| `iterative`           | `IterativeRunner`  | No                  |

Same for getiaction exports — `ActionChunkingRunner` is built‑in. No getiaction import at inference time.

**The exported ONNX/OpenVINO model + built‑in runners = fully self‑contained. No training framework needed at deployment.**

---

## How It Works: Metadata

```yaml
# exports/act_policy/metadata.yaml (getiaction format)
backend: openvino
runner:
  class_path: inferencekit.runners.ActionChunkingRunner
  init_args:
    chunk_size: 100
preprocessors:
  - class_path: physical_ai.pre.ObservationNormalizer
    init_args:
      mean: [0.485, 0.456, 0.406]
```

```json
// exports/pi0/manifest.json (LeRobot format)
{
  "format": "policy_package",
  "policy": { "kind": "iterative" },
  "iterative": { "num_steps": 10, "scheduler": "euler" },
  "artifacts": { "onnx": "model.onnx" }
}
```

**Two formats, same loading path. Both use built‑in runners. Zero external deps.**

---

## How It Works: Discovery & Runtime

```
model path / URI
       │
       ▼
format detection (metadata.yaml? manifest.json?)
       │
       ▼
built‑in runner + pre/post resolved (or external class_path if exotic)
       │
       ▼
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
├── metadata.yaml          # model wiring (runner, pre/post, backend)
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

**Non‑standard execution pattern — write an external plugin:**

1. Create package with Runner / Preprocessor / Postprocessor
2. `pip install -e ./my_plugin`
3. Point `class_path` in `metadata.yaml` at your classes
4. Run: `phyai run --model ./exports/my_model --robot robot.yaml`

**Plugin ownership:** Your team writes it, installs it, maintains it. Your dependencies are your problem. The framework loads it. No PR to our repos needed.

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

Built‑in runners and pre/post cover most cases. **Subclass when orchestration itself must change:**

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

| Milestone   | What happens                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Extract inferencekit from getiaction. Separate package, same functionality.                                                                 |
| **Phase 2** | Build physical‑ai‑framework: format loaders, built‑in runners, observation pipeline, safety, episode orchestration, CLI.                    |
| **Phase 3** | Migrate getiaction deployment to use the framework. GetiAction models load natively via built‑in format loader + ActionChunkingRunner.      |
| **Phase 4** | LeRobot models load natively via built‑in format loader (manifest.json → SinglePassRunner / IterativeRunner). No LeRobot dependency needed. |

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
# Framework only — includes format loaders + built-in runners
# This is all you need for most exported models (GetiAction, LeRobot, custom ONNX)
pip install physical-ai-framework

# With a specific backend
pip install physical-ai-framework[openvino]
pip install physical-ai-framework[onnx-gpu]
```

**No `physical-ai-framework[getiaction]` or `physical-ai-framework[lerobot]` needed.** Built‑in format loaders and runners handle both metadata.yaml and manifest.json natively. No training framework required at deployment time.

External plugins are only for exotic execution patterns — and those are the user's own package (`pip install my-exotic-plugin`).

---

## Key Decisions (Requesting Alignment)

| Decision               | Proposed choice                                          | Why                                                     |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| Extract inference core | inferencekit as separate package                         | Enables vision + robotics + custom domains on same base |
| Interface ownership    | physical‑ai‑framework owns camera/robot ABCs             | One direction. No circular deps.                        |
| Subpackages vs repos   | Subpackages with import linting                          | Versioned together. 3 repos not 5.                      |
| Model loading          | Built‑in format loaders (metadata.yaml, manifest.json)   | Zero external deps for common formats. Day‑1 ready.     |
| Execution patterns     | Built‑in runners (SinglePass, Iterative, ActionChunking) | Most models need zero custom code.                      |
| External plugins       | Only for exotic patterns; user's own package             | User's deps are user's problem. Framework stays thin.   |
| Framework scope        | Observation + safety + episodes + devices                | Not just a loader. Real value.                          |
| Dependency direction   | getiaction → framework → inferencekit                    | Always one‑way. Framework never imports getiaction.     |

---

## Lightweight Guarantee

**`pip install physical-ai-framework` must stay thin.** This is a deployment package — it runs on edge devices, Jetsons, and servers.

| Component                                                            | Where it lives                 | Heavy deps?               |
| -------------------------------------------------------------------- | ------------------------------ | ------------------------- |
| Format loaders (metadata.yaml, manifest.json)                        | Framework (built‑in)           | No — pure file parsing    |
| Built‑in runners (SinglePass, Iterative, ActionChunking)             | Framework (built‑in)           | No — numpy only           |
| Built‑in pre/post (ObservationNormalizer, ActionClamp, TensorResize) | Framework (built‑in)           | No — numpy only           |
| Backend adapters (OpenVINO, ONNX, TensorRT)                          | inferencekit (optional extras) | Only the selected backend |
| External plugins (exotic patterns)                                   | User's own package             | User's problem            |

**What does NOT ship with the framework:**

- No PyTorch
- No training frameworks (getiaction, lerobot, etc.)
- No heavy ML libraries

The framework core is **format loaders + runners + orchestration**. All computation happens in the backend adapter (inferencekit), which installs only the backend you choose.

---

## Summary

1. **The problem is real** — teams duplicate deployment plumbing today
2. **We propose two new packages** — inferencekit (extracted from getiaction) and physical‑ai‑framework (new)
3. **One API, zero external deps** — GetiAction, LeRobot, and custom models deploy the same way, using built‑in format loaders and runners
4. **Framework owns the hard parts** — observation, safety, episodes, devices, validation
5. **Lightweight by design** — `pip install physical-ai-framework` installs only the framework + inferencekit. No torch, no training frameworks. Heavy deps only come from user-owned external plugins for exotic patterns.
6. **No circular dependencies** — one‑directional: getiaction → framework → inferencekit
7. **Incremental path** — extract first, build framework second, migrate third

---

# Q&A

Design docs: `docs/design/deployment/`
