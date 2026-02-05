# Big‑Picture Architecture: InferenceKit + GetiAction + physical‑ai‑framework

## Executive Summary

We want a clean, modular architecture that keeps engineering efficient and gives leadership a clear "product" story.

**Recommendation (balanced and pragmatic):**

- Keep **all core implementation** inside **getiaction**.
- Create a **lightweight deployment repo** (**physical‑ai‑framework**) as a _product shell_ that depends on getiaction.
- Use **inferencekit** as the shared backend‑agnostic inference foundation.

> **Codename**: **physical‑ai‑framework** is the codename for the deployment/inference repo. It provides a thin CLI and packaging layer while reusing getiaction's core implementation.

This achieves:

- **Engineering happiness**: single source of truth, no version skew, faster iteration.
- **Leadership happiness**: a distinct product repo that is easy to position and market.

---

## The Three Layers

### 1) inferencekit (Core Inference)

**Purpose:** Backend‑agnostic `InferenceModel` + runtime adapters/runners.

**Scope:**

- Model loading, prediction, backend abstraction
- OpenVINO / ONNX / TensorRT / Torch Export runtimes
- Runners and callback system

**Non‑Goals:**

- Robotics, cameras, teleoperation, datasets

---

### 2) getiaction (Library + Application)

**Purpose:** End‑to‑end robotics stack.

**Library scope:**

- Robot API + Teleoperation API + Data Collection API
- Policies and preprocessing
- Inference integration (via inferencekit)
- LeRobot compatibility

**Application scope:**

- UI / orchestration / workflow glue

---

### 3) physical‑ai‑framework (Deployment Shell)

**Purpose:** Provide a lightweight deployment package with CLI and docs, while reusing getiaction's implementation.

> **Codename**: **physical‑ai‑framework** is the deployment/inference repo codename.

**Contains:**

- Packaging + branding
- CLI entrypoints (thin wrappers)
- Minimal docs/examples

**Does NOT contain:**

- Core robot / camera / inference implementations
- Duplicated data collection or teleop logic

---

## Why this is the cleanest strategy now

### Benefits for Engineering

- Single codebase for core logic
- No version skew between "core" and "deployment"
- One CI pipeline for correctness
- Faster iteration and easier debugging

### Benefits for Leadership

- Distinct "deployment product" repo exists
- Can be marketed independently
- Lightweight packaging story

---

## How it Works (Dependency Graph)

```
                inferencekit
                      ▲
                      │
                  getiaction
                      ▲
                      │
           physical‑ai‑framework (shell)
```

**Key rule:** physical‑ai‑framework depends on getiaction, not the other way around.

---

## User‑Facing Experience

### CLI via physical‑ai‑framework (Deployment‑only)

```bash
# Example deployment commands
phyai run --config deploy.yaml
phyai serve --model ./exports/policy --robot robot.yaml
```

### Python via getiaction

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

---

## Future‑Proofing (When to Split)

Split physical‑ai‑framework into its own core package **only if**:

- There are **2+ external consumers** beyond getiaction
- You need **independent release cadences**
- You can commit to **contract tests + versioning discipline**

Until then, a thin shell provides the product boundary without engineering risk.

---

## Component Docs

1. **[Teleoperation API](./teleoperation_api.md)** — leader/follower semantics, lifecycle, safety
2. **[Data Collection API](./data_collection_api.md)** — dataset writer, metadata, HF Hub upload
3. **[Deployment Shell](./deployment_shell.md)** — CLI, configuration, and deployment patterns for physical‑ai‑framework

---

## Related Documentation

- **[inferencekit Design](./inferencekit_design.md)** — Generic inference package (Part I)
- **[Robot Interface Design](./robot_interface_design.md)** — Detailed robot interface specification
- **[Camera Interface Design](./camera_interface_design.md)** — Detailed camera interface specification
- **[LeRobot Integration](./inferencekit_lerobot_integration.md)** — LeRobot PolicyPackage integration

---

## Decision Summary

**Recommended now:**

- inferencekit = core inference
- getiaction = implementation and APIs
- physical‑ai‑framework = product shell (thin wrapper)

**Revisit later** only if external consumers and independent release cadence become real.

---

_Last Updated: 2026-02-05_
