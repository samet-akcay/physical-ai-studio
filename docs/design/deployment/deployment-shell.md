# Physical‑AI Inference Engine (physical‑ai‑framework)

Universal physical‑AI inference engine that runs policies from multiple frameworks (getiaction, LeRobot, custom) through a unified API.

---

## Overview

**physical‑ai‑framework** is the universal physical‑AI inference engine. It provides a unified runtime and CLI for running policies from getiaction, LeRobot, or any custom framework — using built‑in format loaders and runners that require zero external dependencies.

**Key Features:**

- Unified API (`InferenceModel`)
- Built‑in format loaders (metadata.yaml, manifest.json) — no training framework imports needed
- Built‑in runners (SinglePassRunner, IterativeRunner, ActionChunkingRunner) — covers common execution patterns
- External plugin support for exotic patterns only (user's own package)
- Configuration-driven workflows
- CLI for edge and server deployment

---

## Personas & Use‑Cases (Quick Start)

### Persona A — GetiAction user (exported getiaction policy)

**Goal:** Run a getiaction policy with the physical‑AI engine.

**Install:**

```bash
pip install physical-ai-framework
```

**Python (policy‑specific API):**

```python
from physical_ai import InferenceModel

policy = InferenceModel("hf://getiaction/act_policy")
action = policy.select_action(observation)
```

**CLI:**

```bash
phyai run --model hf://getiaction/act_policy --robot robot.yaml
```

---

### Persona B — LeRobot user (PolicyPackage)

**Goal:** Run a LeRobot policy package through physical‑ai‑framework.

**Install:**

```bash
pip install physical-ai-framework
```

**Python (policy‑specific API):**

```python
from physical_ai import InferenceModel

policy = InferenceModel("hf://lerobot/pi0")
action = policy.select_action(observation)
```

**CLI:**

```bash
phyai run --model hf://lerobot/pi0 --robot robot.yaml
```

---

### Persona C — Custom model from a standalone GitHub repo

**Goal:** Add support for a new model (custom framework) with an exotic execution pattern not covered by built‑in runners.

**External plugin (editable install):**

```bash
pip install -e ./dreamzero/physical_ai_plugin
```

**Model metadata (in your export directory):**

```yaml
# metadata.yaml
runner:
  class_path: physical_ai_plugin.runner.DreamZeroRunner
preprocessors:
  - class_path: physical_ai_plugin.pre.DreamZeroPre
postprocessors:
  - class_path: physical_ai_plugin.post.DreamZeroPost
```

**Python (unified API):**

```python
from physical_ai import InferenceModel

model = InferenceModel("./exports/dreamzero")
outputs = model(inputs)
```

## Architecture

```
           physical‑ai‑framework (engine)
                        │
          ┌─────────────┼─────────────┐
          │             │             │
   format loaders   built‑in      external plugins
   (metadata.yaml,  runners       (user's own package,
    manifest.json)  (SinglePass,   exotic patterns only)
                    Iterative,
                    ActionChunking)
                        │
                        ▼
                inferencekit (base)
```

**Key principle:** physical‑ai‑framework owns physical‑AI orchestration, hardware lifecycle, and safety. Built‑in format loaders and runners handle common models (getiaction, LeRobot). External plugins are only for exotic execution patterns — user's own package, user's own dependencies. Backend execution lives in inferencekit.

---

## Design Choice: Who Owns the Interfaces?

Two approaches for **camera + robot + inference interfaces**. Both are technically valid. We recommend Option 1.

### Option 1 — physical‑ai‑framework owns the interfaces (recommended)

```
physical‑ai‑framework
  ├─ camera interfaces (physical_ai.camera)
  ├─ robot interfaces  (physical_ai.robot)
  ├─ inferencekit (base)
   └─ format loaders + runners + CLI
          ▲
          │ depends on (hard dependency)
     getiaction (training)
```

**Why this works:**

- **No circular dependency.** getiaction depends on physical‑ai‑framework. physical‑ai‑framework loads models at runtime via format loaders and `class_path` — never imports getiaction at install time. One-directional dependency.
- **Fewer repos (3 instead of 5+).** Only inferencekit, physical‑ai‑framework, and getiaction. Less coordination overhead, simpler CI, fewer version matrices.
- **One package for all hardware interfaces.** Teams install physical‑ai‑framework and get cameras, robots, inference, CLI, safety — everything needed for deployment.
- **Future split is cheap.** Camera/robot interfaces live in clean subpackages (`physical_ai.camera`, `physical_ai.robot`) with no cross-imports. If a vision-only consumer needs camera-api standalone, extract it then. Merging repos later is much harder than splitting.

**Dependency graph:**

```
getiaction → physical‑ai‑framework → inferencekit
                  │
                  ├── physical_ai.camera  (clean subpackage)
                  ├── physical_ai.robot   (clean subpackage)
                   └── physical_ai.engine  (format loaders, runners, CLI, safety)
```

**Condition:** Camera/robot subpackages must have **zero imports** from the rest of physical‑ai‑framework. This is enforced by import linting and makes future extraction trivial.

**Trade-off:** `pip install getiaction` pulls physical‑ai‑framework as a dependency. This is acceptable because getiaction needs hardware interfaces for training (teleoperation, data collection) anyway.

### Option 2 — shared interfaces in separate packages

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
- You need getiaction installable **without** physical-ai-framework (e.g., for CI environments that only run training, no hardware).
- Multiple teams independently maintain camera and robot interfaces with separate release cadences.

**Cost:** 5+ repos instead of 3. Every release requires coordinating versions across camera-api, robot-api, inferencekit, physical-ai-framework, and getiaction. This is real overhead.

**Verdict:** Option 2 is the right choice when the ecosystem is large enough to justify the coordination cost. Start with Option 1 and split when a concrete consumer forces it.

**Important:** Both diagrams show **dependency direction**, not runtime dataflow.

---

## Dependency vs. Runtime Flow (Clarification)

### Dependency (who depends on whom)

```
getiaction → physical‑ai‑framework → inferencekit
```

getiaction depends on physical‑ai‑framework for camera/robot interfaces.
physical‑ai‑framework depends on inferencekit for backend execution.
physical‑ai‑framework loads models at runtime via format loaders (no install-time dependency on getiaction).

### Runtime dataflow (what happens during inference)

```
camera → observation → getiaction preprocessor
     → inferencekit adapter/run → getiaction postprocessor → action → robot
```

---

## Deep‑Dive: How the Engine Works

This section explains **how physical‑ai‑framework resolves format loaders, builds `InferenceModel`, and executes inference**.

**Inference ownership (explicit):**

- **inferencekit** owns backend execution and the base `InferenceModel`
- **Built‑in runners/pre/post** own common execution patterns (SinglePass, Iterative, ActionChunking)
- **External plugins** own exotic execution patterns (user's own package)
- **physical‑ai‑framework** owns orchestration, hardware lifecycle, safety, and the unified API that loads models

### 1) Format Loading & Metadata Resolution

**Discovery order** (first match wins):

1. `metadata.yaml`
2. `metadata.yml`
3. `metadata.json`
4. `manifest.json` (LeRobot PolicyPackage)

**Why this matters:** it lets both **physical‑ai** and **LeRobot** packages work without special‑casing. Both formats are parsed by built‑in format loaders — no external imports needed.

```
model path/URI
     │
     ▼
metadata/manifest detection
     │
     ▼
format loader (built‑in)
     │
     ▼
built‑in runner + pre/post (or external class_path if exotic)
```

### 2) Metadata → Class Instantiation (class_path)

Runners and pre/post are wired by **class_path + init_args** in metadata:

```
metadata.yaml
  ├─ runner.class_path
  ├─ preprocessors[].class_path
  └─ postprocessors[].class_path
```

This allows **external plugins** (editable installs) without upstreaming. For most models, class_paths point to built‑in runners shipped with the framework.

### 3) InferenceModel Runtime Flow

`InferenceModel` composes a pipeline:

```
inputs
  ▼
preprocessors (built‑in or external)
  ▼
runner.run(adapter, inputs)
  ▼
postprocessors (built‑in or external)
  ▼
outputs / action
```

**Notes:**

- `select_action()` wraps `__call__` to return actions.
- Backends execute inside **inferencekit adapters**.

### 4) CLI → Config → Runtime Resolution

```
CLI flags ─┐
           ├─► config resolver ─► runtime config ─► InferenceModel
deploy.yaml ┘

Priority: CLI > config file > defaults
```

### 5) Backend Selection Path

```
model.backend / model.device
          │
          ▼
inferencekit adapter
          │
          ▼
hardware runtime
```

Use the **Backend Selection Guide** to choose values per hardware.

### 6) Extension Points (Built‑in + External)

The framework ships built‑in implementations for common patterns. External plugins extend this for exotic cases:

- **Runners** (execution pattern) — built‑in: SinglePassRunner, IterativeRunner, ActionChunkingRunner
- **Preprocessors** (input shaping) — built‑in: ObservationNormalizer, TensorResize
- **Postprocessors** (output shaping) — built‑in: ActionClamp
- **Callbacks** (instrumentation, safety, logging)

External plugins supply additional runners/pre/post via `class_path` in metadata.

See **[inferencekit Design](./inferencekit.md)** and **[LeRobot Integration](./lerobot.md)** for detailed contracts.

---

## Engine Scope: What physical‑ai‑framework Must Own

physical‑ai‑framework is the **universal physical‑AI inference engine** — not a thin shell that just loads models. To earn that title, it must own the domain-specific concerns that are common to **every** physical‑AI deployment, so teams only supply their model-specific logic.

**The promise:** You write a Runner and a Preprocessor. We handle cameras, robots, safety, episodes, and deployment.

### Engine Capabilities

| Capability                | What the engine provides                                                                                             | What teams still own                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Observation pipeline**  | Camera → observation dict. Standard image capture, buffering, timestamp alignment. Same for every policy.            | Custom observation transforms (e.g., cropping, proprioceptive features) go in preprocessors |
| **Safety runtime**        | Action clamping, velocity limits, workspace bounds, emergency stop. First-class runtime layer, not a callback.       | Domain-specific safety constraints (e.g., force limits for specific robots)                 |
| **Episode orchestration** | Run N episodes, reset between episodes, log results. The control loop is the same for every policy.                  | Episode termination conditions (policy-specific)                                            |
| **Device management**     | Robot connection lifecycle, camera initialization, cleanup on error.                                                 | Robot/camera driver implementations (SDK-specific)                                          |
| **Validation CLI**        | `phyai validate ./exports/my_model` — verify metadata, check class_paths resolve, dry-run pipeline without hardware. | Model-specific validation (e.g., expected input shapes)                                     |
| **Format loading**        | Metadata resolution (metadata.yaml, manifest.json), class_path instantiation, entry point discovery.                 | External plugin code (exotic runners, pre/post processors)                                  |
| **Unified API + CLI**     | `InferenceModel`, `phyai run`, `phyai serve`, config resolution.                                                     | N/A                                                                                         |

### What this means for teams

Without the engine, teams deploying a new model need to:

1. Wire camera capture and observation construction
2. Implement action safety and clamping
3. Write the episode loop (run, reset, log)
4. Manage robot/camera connections and cleanup
5. Build a CLI or script harness

With the engine, teams supply a `metadata.yaml` pointing to built‑in runners and preprocessors. The engine handles everything else. Custom code is only needed for exotic execution patterns.

### Current vs Target State

| Capability                    | Current state                     | Target state                                     |
| ----------------------------- | --------------------------------- | ------------------------------------------------ |
| Format loading + CLI + config | ✓ Designed                        | ✓ Ship as v1                                     |
| Observation pipeline          | Missing                           | v1 — required for "immediate deployment" promise |
| Safety runtime                | Partial (callback)                | v1 — promote to first-class engine layer         |
| Episode orchestration         | Hinted in CLI (`--episodes`)      | v1 — `EpisodeRunner` abstraction                 |
| Device management             | Missing (lives in getiaction app) | v1 — `DeviceManager` for robot/camera lifecycle  |
| Validation CLI                | Missing                           | v1 — `phyai validate`                            |

---

## InferenceModel Hierarchy

Three `InferenceModel` classes exist across the stack. This is intentional — each layer adds domain-specific behavior.

```
inferencekit.InferenceModel          ← base: load model, run forward pass
        │
        │  subclasses / wraps
        ▼
physical_ai.InferenceModel           ← adds: select_action(), reset(), safety,
        │                               observation pipeline, episode management
        │  (getiaction uses this for deployment)
        │
        │  library-only shortcut
        ▼
getiaction.inference.InferenceModel  ← re-exports physical_ai.InferenceModel
                                       for library-only usage without deploying
                                       through the engine
```

**Which one do users import?**

| Use case                           | Import                                            | Why                                                        |
| ---------------------------------- | ------------------------------------------------- | ---------------------------------------------------------- |
| Deploying any policy (recommended) | `from physical_ai import InferenceModel`          | Full engine: safety, observation pipeline, CLI integration |
| Raw inference without engine       | `from inferencekit import InferenceModel`         | Just model + backend, no robotics concerns                 |
| getiaction library scripts         | `from getiaction.inference import InferenceModel` | Convenience re-export; same as physical_ai.InferenceModel  |

---

## Extending InferenceModel (Custom Subclasses)

Built‑in runners, preprocessors, postprocessors, and callbacks cover most customization. **Subclassing `InferenceModel` is for cases the built‑in components cannot express.**

### When built‑in components are enough (do NOT subclass)

| Need                                                  | Plugin             |
| ----------------------------------------------------- | ------------------ |
| Custom forward pass (chunking, iterative, multi-head) | **Runner**         |
| Transform inputs before inference                     | **Preprocessor**   |
| Transform outputs after inference                     | **Postprocessor**  |
| Cross-cutting concerns (timing, logging, safety)      | **Callback**       |
| Different hardware backend                            | **RuntimeAdapter** |

### When to subclass

| Need                                                                    | Why built‑in components aren't enough                 |
| ----------------------------------------------------------------------- | ----------------------------------------------------- |
| Multi-model pipelines (model A feeds model B)                           | Orchestration _between_ pipeline steps changes        |
| Domain-specific convenience API (`warm_up()`, `reset()`, `calibrate()`) | New lifecycle methods the base class doesn't have     |
| Custom model loading that metadata.yaml can't express                   | Loading logic runs before components are instantiated |
| Stateful inference across calls (episode buffer, history window)        | State lives outside the single-call pipeline          |

### Where to subclass

Subclass at the **lowest layer that gives you what you need**:

```
inferencekit.InferenceModel          ← subclass here for non-robotics domains
        │
        ▼
physical_ai.InferenceModel           ← subclass here for robotics with custom orchestration
```

### Example: Multi-model pipeline

```python
from physical_ai import InferenceModel

class PerceptionPolicyModel(InferenceModel):
    """Two-stage: perception model feeds policy model."""

    def __init__(self, policy_path: str, perception_path: str):
        super().__init__(policy_path)
        self.perception = InferenceModel(perception_path)

    def __call__(self, raw_observation):
        features = self.perception(raw_observation)
        return super().__call__(features)
```

### Example: Custom lifecycle methods

```python
from inferencekit import InferenceModel as BaseModel

class WarmableModel(BaseModel):
    """Adds warm-up pass for buffer pre-allocation."""

    def warm_up(self, dummy_input):
        """Run one forward pass to pre-allocate backend buffers."""
        self(dummy_input)

    def benchmark(self, input_data, n_iterations: int = 100):
        """Time N forward passes and return stats."""
        import time
        times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self(input_data)
            times.append(time.perf_counter() - start)
        return {"mean_ms": sum(times) / len(times) * 1000, "n": n_iterations}
```

### Rules for subclasses

1. **Call `super().__init__()`** — metadata loading, component instantiation, and adapter setup happen there.
2. **Don't bypass the pipeline** — override `__call__` only to add steps _around_ `super().__call__()`, not to replace it.
3. **Keep built‑in components working** — a subclass should still load Runner/Pre/Post from metadata. If your subclass ignores metadata, you've left the ecosystem.
4. **Prefer composition over inheritance** — if you need two models, compose two `InferenceModel` instances (like the multi-model example above) rather than merging their logic into one class.

---

## Custom Model Enablement (5‑Minute Guide)

This section answers: **"I have a new model. How do I deploy it with physical‑ai‑framework, right now, without upstreaming anything?"**

### Minimum viable structure

You need two things: an **export directory** with your model and metadata, and a **Python package** with your custom Runner/Preprocessor (if the built-in ones don't suffice).

#### Case 1: Standard single-pass model (no custom code needed)

If your model is a standard ONNX/OpenVINO model that takes inputs and produces outputs in one forward pass, you need **zero custom code**:

```
exports/my_model/
├── model.onnx
└── metadata.yaml
```

```yaml
# metadata.yaml
backend: onnx

runner:
  class_path: inferencekit.runners.SinglePassRunner
  init_args: {}

preprocessors: []
postprocessors: []
```

```python
from physical_ai import InferenceModel

model = InferenceModel("./exports/my_model")
outputs = model(inputs)
```

Done. No plugin package. No entry points. No upstream PR.

#### Case 2: Custom preprocessing or execution pattern

Your model needs custom observation normalization and a non-standard execution pattern (e.g., iterative denoising):

**Step 1 — Create a minimal plugin package:**

```
my_policy_plugin/
├── pyproject.toml
└── my_policy_plugin/
    ├── __init__.py
    ├── preprocessor.py
    └── runner.py
```

```toml
# pyproject.toml
[project]
name = "my-policy-plugin"
version = "0.1.0"
dependencies = ["inferencekit"]
```

```python
# my_policy_plugin/preprocessor.py
from inferencekit.preprocessors import Preprocessor

class MyObservationNormalizer(Preprocessor):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, inputs: dict) -> dict:
        inputs["observation"] = (inputs["observation"] - self.mean) / self.std
        return inputs
```

```python
# my_policy_plugin/runner.py
from inferencekit.runners import InferenceRunner

class MyIterativeRunner(InferenceRunner):
    def __init__(self, num_steps: int = 10):
        self.num_steps = num_steps

    def run(self, adapter, inputs):
        x = inputs["noise"]
        for step in range(self.num_steps):
            inputs["x_t"] = x
            inputs["timestep"] = step / self.num_steps
            v = adapter.predict(inputs)["velocity"]
            x = x + v / self.num_steps
        return {"action": x}

    def reset(self):
        pass
```

**Step 2 — Install locally (editable):**

```bash
pip install -e ./my_policy_plugin
```

**Step 3 — Point metadata at your classes:**

```yaml
# exports/my_model/metadata.yaml
backend: onnx

runner:
  class_path: my_policy_plugin.runner.MyIterativeRunner
  init_args:
    num_steps: 20

preprocessors:
  - class_path: my_policy_plugin.preprocessor.MyObservationNormalizer
    init_args:
      mean: 0.5
      std: 0.2

postprocessors: []
```

**Step 4 — Run:**

```python
from physical_ai import InferenceModel

model = InferenceModel("./exports/my_model")
action = model.select_action(observation)
```

Or via CLI:

```bash
phyai run --model ./exports/my_model --robot robot.yaml
```

**Step 5 — Validate (without hardware):**

```bash
phyai validate ./exports/my_model
# ✓ metadata.yaml found
# ✓ runner class_path resolves: my_policy_plugin.runner.MyIterativeRunner
# ✓ preprocessor class_path resolves: my_policy_plugin.preprocessor.MyObservationNormalizer
# ✓ model.onnx loadable with onnx backend
# ✓ dry-run pipeline: inputs → preprocessors → runner → postprocessors → outputs
```

### Decision flowchart: Runner vs Preprocessor vs Postprocessor

Teams often ask: "Where does my logic go?"

| Question                                                                          | Answer                                                         |
| --------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Does it transform inputs before the model sees them?                              | **Preprocessor** (e.g., normalize observations, resize images) |
| Does it transform outputs after the model produces them?                          | **Postprocessor** (e.g., unnormalize actions, clamp to range)  |
| Does it change _how_ inference runs (multiple passes, action queuing, denoising)? | **Runner** (e.g., iterative diffusion, action chunking)        |
| Is it cross-cutting (logging, timing, safety) and shouldn't modify model code?    | **Callback**                                                   |

---

## Packaging Independence

Teams can **package and distribute their inference pipeline independently**, without any changes to physical‑ai‑framework.

### How it works

The `class_path` mechanism means physical‑ai‑framework never needs to know about your code at install time. It loads your classes dynamically from metadata. As long as your package is installed in the same Python environment, it works.

### Three levels of integration

| Level                                    | What you do                                                          | Who can use it                           | Upstream needed?                                 |
| ---------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------ |
| **Local (editable install)**             | `pip install -e ./my_plugin` + metadata.yaml with class_paths        | You and your team                        | No                                               |
| **Published (PyPI / internal registry)** | `pip install my-policy-plugin` + metadata.yaml                       | Anyone who installs your package         | No                                               |
| **Upstreamed (entry points)**            | Add `[project.entry-points."physical_ai.plugins"]` to pyproject.toml | Auto-discovered by physical-ai-framework | PR to register name, but code stays in your repo |

### Local → Published → Upstream workflow

**Day 1 (local):** Team develops and tests with editable install. No coordination.

```bash
pip install -e ./my_policy_plugin
pip install physical-ai-framework[onnx]
phyai run --model ./exports/my_model --robot robot.yaml
```

**Day 30 (published):** Team publishes to PyPI or internal registry. Other teams can use it.

```bash
pip install my-policy-plugin physical-ai-framework[onnx]
phyai run --model ./exports/my_model --robot robot.yaml
```

**Day 60 (upstreamed, optional):** Team adds entry points so physical‑ai‑framework auto-discovers the plugin.

```toml
# my_policy_plugin/pyproject.toml
[project.entry-points."physical_ai.plugins"]
my_policy = "my_policy_plugin:MyPolicyPlugin"
```

Now `phyai run` auto-discovers the plugin without explicit class_paths in metadata. But this step is **optional** — class_path in metadata always works without entry points.

---

## CLI Reference

### Run Inference

```bash
# Run policy inference on robot
phyai run \
    --model ./exports/act_policy \
    --robot robot.yaml \
    --episodes 10

# With explicit backend
phyai run \
    --model ./exports/act_policy \
    --robot robot.yaml \
    --backend openvino \
    --device CPU
```

### Serve Model

```bash
# Start inference server
phyai serve \
    --model ./exports/act_policy \
    --host 0.0.0.0 \
    --port 8080
```

### Export Policy

```bash
# Export trained policy to deployment format
phyai export \
    --checkpoint ./checkpoints/act_policy \
    --output ./exports \
    --backend onnx
```

---

## Configuration

### Deployment Configuration

```yaml
# deploy.yaml
model:
  path: ./exports/act_policy
  backend: openvino
  device: CPU

robot:
  type: so101
  port: /dev/ttyUSB0
  cameras:
    top:
      type: webcam
      index: 0

inference:
  rate_hz: 30.0
  num_episodes: 10

safety:
  action_min: -1.0
  action_max: 1.0
  velocity_limit: 0.5
```

### Running from Config

```bash
phyai run --config deploy.yaml
```

---

## Python API

### API Entry Points

**InferenceModel** = unified API. Use `__call__` for raw outputs or `select_action()` for observation → action.

```python
from physical_ai import InferenceModel

model = InferenceModel("hf://getiaction/act_policy")
outputs = model(model_inputs)
action = outputs["action"]
```

**Policy‑style usage:**

```python
from physical_ai import InferenceModel

policy = InferenceModel("hf://getiaction/act_policy")
action = policy.select_action(observation)
```

### Native getiaction API (library-only usage)

```python
from getiaction.inference import InferenceModel  # Library-only usage; use physical_ai.InferenceModel for deployment

policy = InferenceModel("./exports/act_policy")
action = policy.select_action(observation)
```

---

## Deployment Patterns

### Pattern 1: Edge Deployment (Jetson)

```yaml
# jetson_deploy.yaml
model:
  path: ./exports/act_policy
  backend: onnx
  device: cuda # TensorRT acceleration

robot:
  type: so101
  port: /dev/ttyUSB0

inference:
  rate_hz: 30.0
  warmup_steps: 10 # Warm up GPU
```

```bash
phyai run --config jetson_deploy.yaml
```

### Pattern 2: Intel CPU Deployment

```yaml
# intel_deploy.yaml
model:
  path: ./exports/act_policy
  backend: openvino
  device: CPU # or GPU for integrated graphics

robot:
  type: so101
  port: /dev/ttyUSB0

inference:
  rate_hz: 30.0
```

### Pattern 3: Server Deployment

```yaml
# server_deploy.yaml
model:
  path: ./exports/act_policy
  backend: onnx
  device: cuda:0

server:
  host: 0.0.0.0
  port: 8080
  workers: 1
```

```bash
phyai serve --config server_deploy.yaml
```

---

## Backend Selection Guide

| Hardware      | Recommended Backend  | Device |
| ------------- | -------------------- | ------ |
| Intel CPU     | `openvino`           | `CPU`  |
| Intel GPU     | `openvino`           | `GPU`  |
| NVIDIA GPU    | `onnx` or `tensorrt` | `cuda` |
| NVIDIA Jetson | `onnx` (TensorRT EP) | `cuda` |
| Apple Silicon | `onnx` (CoreML EP)   | `cpu`  |
| Edge/mobile   | `executorch`         | `cpu`  |

---

## Installation

```bash
# Core package (inference only — includes format loaders + built-in runners)
pip install physical-ai-framework

# With specific backend
pip install physical-ai-framework[openvino]
pip install physical-ai-framework[onnx-gpu]
pip install physical-ai-framework[tensorrt]

# All backends
pip install physical-ai-framework[all]
```

**No `physical-ai-framework[getiaction]` or `physical-ai-framework[lerobot]` needed.** Built‑in format loaders parse metadata.yaml and manifest.json natively. Built‑in runners (SinglePassRunner, IterativeRunner, ActionChunkingRunner) handle common execution patterns. No training framework required at deployment time.

---

## Installation Matrix (What You Get)

| Install command                     | Includes                                               | Excludes          |
| ----------------------------------- | ------------------------------------------------------ | ----------------- |
| `physical-ai-framework`             | Core runtime + CLI + format loaders + built‑in runners | No heavy backends |
| `physical-ai-framework[openvino]`   | Core + OpenVINO runtime                                | No other backends |
| `physical-ai-framework[onnx-gpu]`   | Core + ONNX Runtime (GPU)                              | No other backends |
| `physical-ai-framework[tensorrt]`   | Core + TensorRT                                        | No other backends |
| `physical-ai-framework[executorch]` | Core + ExecuTorch runtime                              | No other backends |
| `physical-ai-framework[all]`        | Core + all backends                                    | —                 |

**Note:** No `[getiaction]` or `[lerobot]` extras exist. Built‑in format loaders and runners handle both frameworks natively. External plugins for exotic patterns are the user's own `pip install`.

---

## What physical‑ai‑framework Contains

**Contains:**

- Unified inference runtime for physical‑AI policies (`InferenceModel`)
- Observation pipeline (camera → observation dict)
- Safety runtime (action clamping, velocity limits, emergency stop)
- Episode orchestration (run N episodes, reset, log)
- Device management (robot/camera connection lifecycle)
- Format loaders (metadata.yaml, manifest.json) and built‑in runners
- CLI entrypoints (`phyai run`, `phyai serve`, `phyai export`, `phyai validate`)
- Configuration loading and validation
- Camera interfaces (`physical_ai.camera`) — clean subpackage, no cross-imports
- Robot interfaces (`physical_ai.robot`) — clean subpackage, no cross-imports

**Does NOT contain:**

- Backend execution (lives in inferencekit)
- Vision model wrappers and preprocessing (lives in model_api)
- Training code (lives in getiaction)
- Exotic policy-specific runners, pre/postprocessors (lives in external plugins — user's own package)

---

## Related Documentation

- **[Strategy](../strategy.md)** - Big-picture architecture
- **[inferencekit Design](./inferencekit.md)** - Core inference package
- **[Robot Interface Design](../library/robot-interface.md)** - Robot interface specification

---

_Last Updated: 2026-02-11_
