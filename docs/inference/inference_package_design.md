# Universal Inference Package Design (`inferencekit`)

A backend- and domain-agnostic inference framework that provides a single, consistent API across:

- Domains: robotics (LeRobot), computer vision (Geti ecosystem), and beyond
- Inference patterns: single-pass, iterative (diffusion/flow matching), stateful (action chunking, KV cache), and multi-stage (e.g., SAM)
- Backends: ONNX Runtime, OpenVINO, TensorRT, PyTorch (eager / compile / export), etc.

---

## Executive Summary

### What this is

`inferencekit` is a **core inference package** that standardizes how we load exported models, run predictions, add instrumentation/safety hooks, and swap execution backends.

### Why this exists

Across ML ecosystems, we repeatedly rebuild the same glue:

- Per-domain wrappers that slightly differ in predict/reset/lifecycle semantics
- Per-backend integration code spread across projects
- Ad-hoc handling for models that are not “one forward pass” (diffusion, action chunking, multi-stage architectures)
- Instrumentation (timing/logging/telemetry) that is inconsistent and hard to reuse

`inferencekit` centralizes these concerns behind a small set of stable primitives.

### Non-goals

- Training or fine-tuning interfaces (these remain in domain packages)
- Defining a single export format for all models (we consume existing exports + metadata)
- Becoming a generic data pipeline/ETL system

---

## Quickstart: High-Level API

The goal is that **most users never need to learn about backends, runners, or adapters**.

### Minimal usage (90%)

```python
from inferencekit import InferenceModel

model = InferenceModel.load("./exports/my_model")
outputs = model.predict(inputs)
```

### Easy customization (backend, device, callbacks)

```python
from inferencekit import InferenceModel
from inferencekit.callbacks import TimingCallback, LoggingCallback

model = InferenceModel.load(
    "./exports/my_model",
    backend="onnx",
    device="cuda",
    callbacks=[TimingCallback(), LoggingCallback()],
)
outputs = model.predict(inputs)
```

### Override runner parameters without touching internals

```python
# Autodetect the runner from metadata, but override its parameters
model = InferenceModel.load(
    "./exports/diffusion_policy",
    num_inference_steps=50,
    scheduler_type="ddim",
)
```

### Full control (explicit runner)

```python
from inferencekit import InferenceModel
from lerobot.inference.runners import DiffusionRunner

model = InferenceModel.load(
    "./exports/diffusion_policy",
    runner=DiffusionRunner(num_inference_steps=50, scheduler_type="ddim"),
)
outputs = model.predict(obs)
```

### Lifecycle for stateful models

```python
with InferenceModel.load("./exports/robot_policy") as model:
    model.reset()  # start episode/sequence
    for obs in observations:
        action = model.predict(obs)
```

---

## Mental Model: 4 Core Concepts

`inferencekit` is built around four composable primitives:

1. **InferenceModel** – user-facing orchestrator (load/predict/reset/close)
2. **InferenceRunner** – defines _how inference runs_ (single-pass, iterative, stateful, multi-stage, or custom)
3. **RuntimeAdapter** – executes _one forward pass_ on a specific backend (ONNX/OpenVINO/TensorRT/Torch)
4. **Callback** – cross-cutting hooks (timing/logging/safety/checkpointing)

### Call flow

```
User → InferenceModel.predict() → InferenceRunner.run() → RuntimeAdapter.predict()
```

- **Runner orchestrates.** It may call the adapter once (single-pass) or many times (diffusion loop), and it may preserve state across calls (action chunking, KV cache).
- **Adapter executes.** It is intentionally stateless: a thin wrapper over the underlying runtime.
- **Callbacks observe/modify.** They can record metrics, enforce safety limits, log predictions, or checkpoint state.

---

## Extensibility (How users customize)

Customization is intended to be incremental and approachable:

### 1) Add callbacks (recommended first extension point)

Callbacks provide Lightning-compatible hooks to instrument or modify prediction behavior.

Examples:

- Performance timing / profiling
- Prediction logging
- Safety clamps for robotics actions
- State checkpointing

### 2) Add preprocessors / postprocessors

- **Preprocessors:** input transformations (resize/normalize/tokenize)
- **Postprocessors:** output transformations (NMS, thresholding, label mapping, mask resizing)

These should be cleanly separable from how inference runs.

### 3) Add adapters (new backend)

Implement `RuntimeAdapter` for a new runtime backend.

### 4) Add runners (only when inference pattern is non-trivial)

Runners exist only to support models that are not a single forward pass.

---

## When is a Custom Runner Needed?

A **Runner** defines the **inference algorithm**: HOW the model runs, not what happens to outputs.

### Use `SinglePassRunner` (default) when:

```python
# Simple pattern: preprocess → forward → postprocess
outputs = model(inputs)
```

| Model Type             | Runner     | Why                                       |
| ---------------------- | ---------- | ----------------------------------------- |
| Classification         | SinglePass | One forward pass                          |
| Detection (YOLO, DETR) | SinglePass | One forward pass (NMS is post-processing) |
| Segmentation           | SinglePass | One forward pass                          |
| Anomaly Detection      | SinglePass | One forward pass                          |
| ACT (single action)    | SinglePass | One forward pass                          |

### Use a custom runner when:

| Pattern                      | Runner               | Why                                           |
| ---------------------------- | -------------------- | --------------------------------------------- |
| **Iterative loops**          | DiffusionRunner      | 10–100 denoising steps                        |
| **Stateful across calls**    | ActionChunkingRunner | Action queue persists between `predict()`     |
| **Multi-stage architecture** | SAMRunner            | Image encoder → Prompt encoder → Mask decoder |
| **KV cache management**      | FlowMatchingRunner   | Cache prefix embeddings, run loop             |
| **Chained models**           | ChainRunner          | Detection → Classification chain              |

### Visual summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SinglePassRunner                                 │
│                    (Default - 90% of models)                            │
│                                                                         │
│    inputs → preprocess() → adapter.predict() → postprocess() → outputs  │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      Specialized Runners                                  │
│              (Only when inference pattern is different)                  │
│                                                                          │
│  DiffusionRunner:                                                        │
│    x = noise                                                             │
│    for t in timesteps:                                                   │
│        x = adapter.predict(x, t)  ← multiple calls                       │
│        x = scheduler.step(x)                                             │
│                                                                          │
│  ActionChunkingRunner:                                                   │
│    if action_queue not empty:                                            │
│        return action_queue.pop()  ← stateful, may skip model call        │
│    actions = adapter.predict(obs)                                        │
│    action_queue.extend(actions)                                          │
│                                                                          │
│  SAMRunner:                                                              │
│    image_emb = image_encoder.predict(image)     ← stage 1                │
│    prompt_emb = prompt_encoder.predict(points)  ← stage 2                │
│    masks = decoder.predict(image_emb, prompt_emb) ← stage 3              │
│                                                                          │
│  FlowMatchingRunner:                                                     │
│    kv_cache = encoder.predict(prefix)  ← cached across calls             │
│    for step in flow_steps:                                               │
│        x = expert.predict(x, kv_cache)                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core API Design (Conceptual)

### 1) `InferenceModel` (main entry point)

`InferenceModel` is the orchestrator (similar to Lightning’s `Trainer`, but for inference). It:

- Loads metadata from an export directory
- Creates a runtime adapter (backend-specific)
- Selects/instantiates a runner (explicit or from metadata)
- Manages lifecycle (`predict`, `predict_batch`, `reset`, `close`)
- Dispatches callback hooks

**API tiers:**

- **Tier 1:** `InferenceModel.load(path)` + `predict()`
- **Tier 2:** pass runner params as kwargs to override metadata defaults
- **Tier 3:** pass an explicit `runner=...`

### 2) `InferenceRunner` (the algorithm)

A runner defines HOW inference runs:

- SinglePassRunner: one forward pass
- DiffusionRunner: iterative denoising steps
- ActionChunkingRunner: stateful action queue
- SAMRunner: multi-stage inference

Runners may be stateful and should implement `reset()` when appropriate.

### 3) `Callback` (Lightning-compatible hooks)

Callbacks provide a stable extension mechanism for cross-cutting behavior.

Key properties:

- Hook order is explicit
- Hooks can optionally modify inputs/outputs by returning replacements
- Iterative runners can call per-step hooks

### 4) `RuntimeAdapter` (backend abstraction)

Adapters implement a minimal contract:

- `load(model_path)`
- `predict(inputs)` for a single forward pass
- Optional introspection (`input_names`, `output_names`, `input_shapes`, `output_shapes`)

Adapters should be intentionally stateless; orchestration belongs in runners.

---

## Metadata Format (`class_path` + `init_args`)

Following `jsonargparse` conventions (already used in `getiaction`), metadata uses a `class_path` + `init_args` structure.

### YAML or JSON

Both formats are supported. YAML is recommended for human-edited files.

**Loading priority:**

1. `metadata.yaml`
2. `metadata.yml`
3. `metadata.json`

### Example (YAML, recommended)

```yaml
model_type: diffusion_policy
backend: onnx

runner:
  class_path: lerobot.inference.runners.DiffusionRunner
  init_args:
    num_inference_steps: 100
    scheduler_type: ddpm

postprocessor:
  class_path: getitune.inference.postprocessors.DetectionPostprocessor
  init_args:
    score_threshold: 0.5
    nms_threshold: 0.45
    labels:
      - cat
      - dog
      - bird

adapter:
  class_path: inferencekit.adapters.ONNXAdapter
  init_args:
    providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
```

### Benefits

1. **Self-describing:** metadata reconstructs objects
2. **No registry required:** can load via dynamic import
3. **Consistent:** runner/postprocessor/adapter all share the same schema
4. **Compatible:** aligns with existing tooling
5. **Overridable:** user kwargs override `init_args` at load time

---

## Package Structure

```
# ==========================================================================
# CORE PACKAGE (shared across all domains)
# ==========================================================================
inferencekit/
├── __init__.py
├── model.py                    # InferenceModel - main entry point
├── runners/                    # Only for NON-TRIVIAL inference patterns
│   ├── __init__.py             # Optional registry + get_runner()
│   ├── base.py                 # InferenceRunner ABC
│   └── single_pass.py          # Default - covers 90% of models
├── postprocessors/             # Output transformation (NMS, thresholding, etc.)
│   ├── __init__.py
│   ├── base.py                 # Postprocessor ABC
│   ├── nms.py                  # Non-max suppression
│   ├── masks.py                # Mask resizing, thresholding
│   └── classification.py       # Softmax, top-k
├── preprocessors/              # Input transformation (resize, normalize, etc.)
│   ├── __init__.py
│   ├── base.py                 # Preprocessor ABC
│   └── image.py                # Image preprocessing utilities
├── callbacks/
│   ├── __init__.py
│   ├── base.py                 # Callback ABC with Lightning-compatible hooks
│   ├── timing.py               # Performance profiling
│   ├── logging.py              # Prediction logging
│   └── checkpoint.py           # State checkpointing
├── adapters/
│   ├── __init__.py
│   ├── base.py                 # RuntimeAdapter ABC
│   ├── onnx.py                 # ONNX Runtime
│   ├── openvino.py             # OpenVINO
│   ├── tensorrt.py             # TensorRT
│   ├── torch.py                # PyTorch eager
│   └── torch_compile.py        # torch.compile / Torch Export IR
├── io/
│   ├── __init__.py
│   ├── metadata.py             # Metadata loading/saving
│   └── tensors.py              # Tensor I/O utilities
└── utils/
    ├── __init__.py
    ├── instantiate.py          # class_path + init_args instantiation
    ├── device.py               # Device detection/management
    └── dtype.py                # Dtype handling

# ==========================================================================
# LEROBOT (Robotics - Open Source)
# Custom runners needed - inference patterns differ from single forward pass
# ==========================================================================
lerobot/
├── inference/
│   ├── __init__.py             # Domain-specific exports
│   ├── runners/
│   │   ├── action_chunking.py  # Stateful action queue management
│   │   ├── diffusion.py        # Iterative denoising loop (10–100 steps)
│   │   └── flow_matching.py    # KV cache + flow matching loop
│   └── callbacks/
│       ├── action_safety.py    # Action clamping, joint limits
│       └── episode_logging.py  # Episode recording

# ==========================================================================
# GETI ECOSYSTEM
# ==========================================================================

# --- getitune: Fine-tuning for vision tasks ---
# NO custom runners - all single forward pass
# Only postprocessors and result types
getitune/
├── inference/
│   ├── __init__.py
│   ├── postprocessors/
│   │   ├── detection.py        # NMS, box decoding, score filtering
│   │   ├── segmentation.py     # Mask resizing, class mapping
│   │   └── classification.py   # Top-k, label mapping
│   ├── results/                # Result dataclasses
│   │   ├── detection.py        # DetectionResult, Detection
│   │   ├── segmentation.py     # SegmentationResult
│   │   └── classification.py   # ClassificationResult
│   └── callbacks/
│       ├── visualization.py    # Draw boxes, masks on images
│       └── metrics.py          # mAP, IoU, accuracy

# --- getiprompt: Zero/few-shot prompting ---
# SAM needs custom runner (multi-stage), others likely don't
getiprompt/
├── inference/
│   ├── __init__.py
│   ├── runners/
│   │   ├── sam.py              # Multi-stage: image enc → prompt enc → decoder
│   │   └── sam2_video.py       # Video tracking with memory bank
│   ├── postprocessors/
│   │   └── grounding.py        # Text-to-box post-processing
│   └── callbacks/
│       ├── prompt_logging.py   # Log prompts and responses
│       └── interactive.py      # Interactive prompting support

# --- getiinspect: Anomaly detection (Anomalib-based) ---
# NO custom runners - all single forward pass
getiinspect/
├── inference/
│   ├── __init__.py
│   ├── postprocessors/
│   │   └── anomaly.py          # Thresholding, normalization
│   ├── results/
│   │   └── anomaly.py          # AnomalyResult
│   └── callbacks/
│       ├── threshold_tuning.py # Adaptive thresholding
│       ├── heatmap.py          # Anomaly heatmap visualization
│       └── metrics.py          # AUROC, F1, PRO

# --- getiaction: VLA / Robotics ---
# Reuses LeRobot runners
getiaction/
├── inference/
│   ├── __init__.py             # Imports from lerobot.inference
│   └── callbacks/
│       ├── robot_safety.py     # Robot-specific safety
│       └── telemetry.py        # Robot telemetry logging
```

---

## Appendix: Package Naming

### Current state: `model_api`

The current `model_api` package provides model inference interfaces, but the name does not emphasize inference and can be interpreted as “interfaces only”, rather than an execution toolkit.

### Proposed name: `inferencekit`

| Aspect              | `inferencekit`                 | Benefit                                       |
| ------------------- | ------------------------------ | --------------------------------------------- |
| **Clarity**         | “Toolkit for inference”        | Clear purpose                                 |
| **Scope**           | “Kit” implies composable parts | Matches design (runners, adapters, callbacks) |
| **Discoverability** | Specific to inference          | Easy to find                                  |
| **Association**     | Backend-agnostic               | Neutral, professional                         |

### Alternatives considered

| Name               | Pros                            | Cons                        | Verdict         |
| ------------------ | ------------------------------- | --------------------------- | --------------- |
| **`inferencekit`** | Clear, professional, composable | Slightly long               | ✅ Recommended  |
| **`model_api`**    | Existing, familiar              | Doesn’t highlight inference | Maybe (current) |
| **`inferencelib`** | Classic “-lib” suffix           | Generic                     | Maybe           |
| **`inferencer`**   | Clean                           | Sounds like a class name    | No              |
| **`runtime`**      | Industry standard               | Overloaded                  | No              |

---

## Notes / Open Questions

(Keep this section short; add items only as decisions are needed.)

- Do we want a **registry** (short names) in addition to `class_path`? This can remain a future enhancement.
- Should `InferenceModel.load()` be the primary constructor, or should `InferenceModel(...)` be encouraged? (Both are compatible.)
