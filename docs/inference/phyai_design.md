# phyai Design Document

**Robotics-focused deployment framework built on top of inferencekit.**

---

## Table of Contents

- [Overview](#overview)
  - [Purpose](#purpose)
  - [Relationship to inferencekit](#relationship-to-inferencekit)
  - [Design Goals](#design-goals)
- [Architecture](#architecture)
  - [Layered Architecture](#layered-architecture)
  - [Package Structure](#package-structure)
- [Robotics-Specific Runners](#robotics-specific-runners)
  - [IterativeRunner](#iterativerunner)
  - [ActionChunkingRunner](#actionchunkingrunner)
- [Robotics-Specific Callbacks](#robotics-specific-callbacks)
- [Multi-Framework Support](#multi-framework-support)
  - [geti-action Integration](#geti-action-integration)
  - [LeRobot Integration](#lerobot-integration)
  - [Framework Extension Pattern](#framework-extension-pattern)
- [Camera Interface](#camera-interface)
- [Robot Interface](#robot-interface)
- [End-to-End Deployment Patterns](#end-to-end-deployment-patterns)
  - [Minimal Inference (inferencekit Only)](#minimal-inference-inferencekit-only)
  - [Camera + Inference](#camera--inference)
  - [Full Robot Control Loop](#full-robot-control-loop)
  - [Edge Deployment Considerations](#edge-deployment-considerations)
  - [Safety Considerations](#safety-considerations)
- [Compatibility with LeRobot Export](#compatibility-with-lerobot-export)
  - [Shared Design Principles](#shared-design-principles)
  - [PolicyPackage Format](#policypackage-format)
  - [Runtime Compatibility](#runtime-compatibility)
  - [Maintenance Strategy](#maintenance-strategy)
- [API Reference](#api-reference)
- [Related Documents](#related-documents)

---

## Overview

### Purpose

The **phyai** framework is a robotics-focused deployment framework that provides:

- Camera interface for visual observations
- Robot interface for action execution
- Robotics-specific inference runners
- Multi-framework support (geti-action, LeRobot)
- Safety callbacks for robot deployment

This is the first open-source, edge-focused physical AI inference framework.

### Relationship to inferencekit

phyai **builds on top of** inferencekit:

```python
# phyai uses inferencekit for core inference
from inferencekit import InferenceModel
from phyai.cameras import RealSense
from phyai.robots import SO101

# InferenceModel comes from inferencekit
policy = InferenceModel.load("./exports/act_policy")

# Cameras and robots come from phyai
camera = RealSense(fps=30)
robot = SO101.from_config("robot.yaml")
```

**Key separation:**

| Component                   | Package      |
| --------------------------- | ------------ |
| `InferenceModel`            | inferencekit |
| `RuntimeAdapter`            | inferencekit |
| `SinglePassRunner`          | inferencekit |
| `TimingCallback`            | inferencekit |
| `IterativeRunner`           | phyai        |
| `ActionChunkingRunner`      | phyai        |
| `ActionSafetyCallback`      | phyai        |
| Camera interface            | phyai        |
| Robot interface             | phyai        |
| geti-action/LeRobot plugins | phyai        |

### Design Goals

| Goal                         | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| **G1: Multi-Framework**      | Native integration with geti-action, LeRobot, extensible to others |
| **G2: Hardware Abstraction** | Camera and robot interfaces across vendors                         |
| **G3: Edge Optimization**    | Designed for resource-constrained devices                          |
| **G4: Production Safety**    | Action clamping, logging, emergency stops                          |
| **G5: Minimal Overhead**     | Thin layer over generic inference package                          |

---

## Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              phyai                                          │
│                     (Physical AI Framework)                                 │
│                                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────────────┐  │
│  │    Camera     │  │    Robot      │  │   Robotics Components           │  │
│  │   Interface   │  │   Interface   │  │   - IterativeRunner             │  │
│  │               │  │               │  │   - ActionChunkingRunner        │  │
│  └───────────────┘  └───────────────┘  │   - ActionSafetyCallback        │  │
│                                        │   - EpisodeLoggingCallback      │  │
│                                        └─────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Framework Integrations                           │    │
│  │                                                                     │    │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │    │
│  │   │ geti-action │   │   LeRobot   │   │   Future Frameworks     │   │    │
│  │   │   Plugin    │   │   Plugin    │   │       (extensible)      │   │    │
│  │   └─────────────┘   └─────────────┘   └─────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ depends on
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          inferencekit                                       │
│                   (Generic Inference Package)                               │
│                                                                             │
│  InferenceModel │ RuntimeAdapters │ SinglePassRunner │ Callbacks            │
│                                                                             │
│  Backends: OpenVINO │ ONNX Runtime │ TensorRT │ Torch Export IR             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Package Structure

```text
phyai/                                       # Physical AI Framework
├── __init__.py
├── runners/
│   ├── __init__.py
│   ├── iterative.py                         # IterativeRunner (diffusion, flow)
│   └── action_chunking.py                   # ActionChunkingRunner
├── callbacks/
│   ├── __init__.py
│   ├── safety.py                            # ActionSafetyCallback
│   └── episode_logging.py                   # EpisodeLoggingCallback
├── cameras/                                 # Camera interface
│   ├── __init__.py
│   ├── base.py                              # Camera ABC
│   ├── webcam.py
│   ├── realsense.py
│   ├── basler.py
│   ├── genicam.py
│   ├── ipcam.py
│   ├── screen.py
│   ├── video.py                             # VideoFile
│   └── folder.py                            # ImageFolder
├── robots/                                  # Robot interface
│   ├── __init__.py
│   ├── base.py                              # Robot ABC
│   └── lerobot/                             # LeRobot robot wrappers
│       ├── __init__.py
│       ├── universal.py                     # LeRobotRobot
│       ├── so101.py
│       └── aloha.py
└── plugins/
    ├── __init__.py
    ├── getiaction.py                        # geti-action metadata support
    └── lerobot.py                           # LeRobot PolicyPackage support
```

---

## Robotics-Specific Runners

These runners are specific to robotics and belong in phyai.

### IterativeRunner

For diffusion policies with iterative denoising:

```python
class IterativeRunner(InferenceRunner):
    """Runner for iterative inference (diffusion).

    Performs multiple forward passes with noise scheduling.
    Used by: Diffusion Policy.
    """

    def __init__(
        self,
        num_steps: int = 10,
        scheduler: str = "euler",
        timestep_spacing: str = "linear",
    ):
        self.num_steps = num_steps
        self.scheduler = scheduler
        self.timestep_spacing = timestep_spacing

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Initialize from noise
        x_t = np.random.randn(*action_shape).astype(np.float32)

        # Generate timesteps
        timesteps = self._generate_timesteps()

        # Iterative denoising
        for t in timesteps:
            step_inputs = {**inputs, "x_t": x_t, "timestep": np.array([t])}
            v_t = adapter.predict(step_inputs)["v_t"]
            x_t = self._step(x_t, v_t, t)

        return {"action": x_t}

    def reset(self) -> None:
        pass  # Stateless
```

### TwoPhaseRunner

For VLA policies with encoder caching (PI0, SmolVLA):

```python
class TwoPhaseRunner(InferenceRunner):
    """Runner for two-phase VLA inference.

    Phase 1: Encoder processes observations → KV cache
    Phase 2: Iterative denoising using cached KV

    Used by: PI0, SmolVLA, and other VLA models.
    """

    def __init__(
        self,
        num_steps: int = 10,
        encoder_artifact: str = "onnx_encoder",
        denoise_artifact: str = "onnx_denoise",
        num_layers: int = 18,
        num_kv_heads: int = 8,
        head_dim: int = 256,
    ):
        self.num_steps = num_steps
        self.encoder_artifact = encoder_artifact
        self.denoise_artifact = denoise_artifact
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Phase 1: Encode observations → KV cache
        encoder_outputs = adapter.predict_encoder(inputs)
        kv_cache = encoder_outputs["kv_cache"]
        prefix_mask = encoder_outputs["prefix_pad_mask"]

        # Phase 2: Iterative denoising with cached KV
        x_t = np.random.randn(*action_shape).astype(np.float32)

        for t in self._generate_timesteps():
            denoise_inputs = {
                "x_t": x_t,
                "timestep": np.array([t]),
                "kv_cache": kv_cache,
                "prefix_pad_mask": prefix_mask,
            }
            v_t = adapter.predict_denoise(denoise_inputs)["v_t"]
            x_t = self._step(x_t, v_t, t)

        return {"action": x_t}

    def reset(self) -> None:
        pass  # Stateless (KV cache is per-inference, not persistent)
```

### ActionChunkingRunner

For policies that predict action chunks (ACT, etc.):

```python
class ActionChunkingRunner(InferenceRunner):
    """Runner with action queue for chunked policies.

    Predicts chunk_size actions, returns one at a time.
    Used by: ACT, temporal ensemble policies.
    """

    def __init__(self, chunk_size: int = 100, n_action_steps: int | None = None):
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps or chunk_size
        self._action_queue: deque = deque()

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        if len(self._action_queue) > 0:
            return {"action": self._action_queue.popleft()}

        # Run model to get action chunk
        outputs = adapter.predict(inputs)
        actions = outputs["action"]  # Shape: (chunk_size, action_dim)

        # Queue actions (except first, which we return)
        self._action_queue.extend(actions[1:self.n_action_steps])
        return {"action": actions[0]}

    def reset(self) -> None:
        self._action_queue.clear()
```

---

## Robotics-Specific Callbacks

```python
class ActionSafetyCallback(Callback):
    """Clamp actions to safe ranges."""

    def __init__(
        self,
        action_min: np.ndarray | float = -1.0,
        action_max: np.ndarray | float = 1.0,
        velocity_limit: float | None = None,
    ):
        self.action_min = action_min
        self.action_max = action_max
        self.velocity_limit = velocity_limit
        self._last_action = None

    def on_predict_end(self, outputs: dict) -> dict:
        action = outputs["action"]

        # Clamp to range
        action = np.clip(action, self.action_min, self.action_max)

        # Velocity limiting
        if self.velocity_limit and self._last_action is not None:
            delta = action - self._last_action
            delta = np.clip(delta, -self.velocity_limit, self.velocity_limit)
            action = self._last_action + delta

        self._last_action = action.copy()
        outputs["action"] = action
        return outputs

    def on_reset(self) -> None:
        self._last_action = None


class EpisodeLoggingCallback(Callback):
    """Log episode data for replay/debugging."""

    def __init__(self, log_dir: Path, log_observations: bool = True):
        self.log_dir = Path(log_dir)
        self.log_observations = log_observations
        self._episode_data = []
        self._episode_count = 0

    def on_predict_end(self, outputs: dict, inputs: dict | None = None) -> dict:
        step_data = {"action": outputs["action"].tolist()}
        if self.log_observations and inputs:
            step_data["observation"] = {k: v.tolist() for k, v in inputs.items()}
        self._episode_data.append(step_data)
        return outputs

    def on_reset(self) -> None:
        if self._episode_data:
            self._save_episode()
        self._episode_data = []
        self._episode_count += 1
```

---

## Multi-Framework Support

Framework integrations live in phyai via plugins.

### geti-action Integration

geti-action policies export metadata compatible with inferencekit. phyai provides a plugin that maps geti-action metadata to appropriate runners:

```python
# Training (geti-action)
from getiaction.policies.act import ACT
from getiaction.train import Trainer

policy = ACT(...)
trainer.fit(policy, datamodule)
policy.export("./exports", backend="openvino")

# Inference (uses inferencekit + phyai runner)
from inferencekit import InferenceModel

model = InferenceModel.load("./exports")  # Plugin auto-selects runner
action = model.predict(observation)
```

**Supported geti-action policies:**

| Policy    | Runner               | Notes                         |
| --------- | -------------------- | ----------------------------- |
| ACT       | ActionChunkingRunner | Action queue management       |
| Diffusion | IterativeRunner      | DDPM/DDIM scheduling          |
| Pi0       | TwoPhaseRunner       | VLA with encoder caching      |
| SmolVLA   | TwoPhaseRunner       | VLA with encoder caching      |
| GR00T     | ActionChunkingRunner | Single-pass with action queue |

### LeRobot Integration

LeRobot integration via plugin, no circular dependencies. phyai reads LeRobot's `manifest.json` and maps inference patterns to appropriate runners.

**Note:** LeRobot uses structural detection to determine inference type (based on manifest structure, not explicit `kind` field).

```python
# phyai/plugins/lerobot.py

@register_format("policy_package")
class LeRobotPlugin:
    """Plugin for loading LeRobot PolicyPackages."""

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a LeRobot PolicyPackage."""
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text())
        return manifest.get("format") == "policy_package"

    @staticmethod
    def load(path: Path, **kwargs) -> InferenceModel:
        """Load PolicyPackage into InferenceModel."""
        manifest = json.loads((path / "manifest.json").read_text())

        # Determine runner from manifest structure
        runner = _create_runner_from_manifest(manifest, **kwargs)

        # Create adapter from artifacts
        backend = kwargs.get("backend") or _select_default_backend(manifest)
        adapter = get_adapter(backend)(path / manifest["artifacts"][backend])

        return InferenceModel(adapter=adapter, runner=runner, ...)


def _create_runner_from_manifest(manifest: dict, **kwargs) -> InferenceRunner:
    """Map LeRobot manifest to phyai runner using structural detection."""
    inference = manifest.get("inference")
    action = manifest.get("action", {})

    if inference is None:
        # Single-pass model (ACT, VQ-BeT, Groot)
        # May still need action chunking based on action spec
        chunk_size = action.get("chunk_size", 1)
        n_action_steps = action.get("n_action_steps", chunk_size)
        if chunk_size > 1:
            return ActionChunkingRunner(
                chunk_size=chunk_size,
                n_action_steps=n_action_steps,
            )
        return SinglePassRunner()

    elif "encoder_artifact" in inference:
        # Two-phase VLA (PI0, SmolVLA)
        return TwoPhaseRunner(
            num_steps=inference.get("num_steps", 10),
            encoder_artifact=inference["encoder_artifact"],
            denoise_artifact=inference["denoise_artifact"],
            num_layers=inference.get("num_layers"),
            num_kv_heads=inference.get("num_kv_heads"),
            head_dim=inference.get("head_dim"),
        )

    elif "scheduler" in inference:
        # Iterative denoising (Diffusion)
        return IterativeRunner(
            num_steps=kwargs.get("num_steps", inference.get("num_steps", 10)),
            scheduler=kwargs.get("scheduler", inference.get("scheduler", "euler")),
            timestep_spacing=inference.get("timestep_spacing", "linear"),
        )

    else:
        raise ValueError("Cannot determine inference type from manifest")
```

### Framework Extension Pattern

Adding support for new frameworks:

```python
# my_framework_plugin.py
from inferencekit.plugins import register_format

@register_format("my_format")
class MyFrameworkPlugin:
    @staticmethod
    def detect(path: Path) -> bool:
        return (path / "my_config.json").exists()

    @staticmethod
    def load(path: Path, **kwargs) -> InferenceModel:
        # Load and return InferenceModel with appropriate runner
        ...
```

---

## Camera Interface

phyai provides a unified camera interface for visual observations. The interface abstracts differences between camera hardware and provides consistent frame acquisition.

**Key Features:**

- Unified `Camera` ABC across all hardware
- Multi-consumer support (multiple processes reading same camera)
- Context manager for safe resource management
- Callback system for push-based frame delivery
- Capability mixins (depth, PTZ, etc.)

**Quick Example:**

```python
from phyai.cameras import Webcam, RealSense

# Simple camera usage
with Webcam(index=0, fps=30, width=640, height=480) as camera:
    frame = camera.read()

# Depth camera
with RealSense(serial="123456") as camera:
    rgb = camera.read()
    depth = camera.read_depth()
```

**Supported Cameras:**

| Camera        | Hardware            | Backend      |
| ------------- | ------------------- | ------------ |
| `Webcam`      | USB cameras         | OpenCV       |
| `RealSense`   | Intel depth cameras | pyrealsense2 |
| `Basler`      | Industrial cameras  | pypylon      |
| `Genicam`     | GenICam devices     | harvesters   |
| `IPCam`       | Network cameras     | RTSP/OpenCV  |
| `Screen`      | Desktop capture     | mss          |
| `VideoFile`   | Video playback      | OpenCV       |
| `ImageFolder` | Image sequences     | OpenCV       |

**For detailed camera interface design, see:** [Camera Interface Design](../camera_interface_design.md)

---

## Robot Interface

phyai provides a unified robot interface for action execution and state reading.

> **Note:** This section provides a high-level overview. Detailed implementation design is subject to further discussion due to scalability concerns around per-robot configuration complexity.

**Key Features:**

- Framework-agnostic (works with geti-action, LeRobot, custom pipelines)
- Standard types at core (`dict`, `np.ndarray`)
- Optional format conversion to framework-specific formats
- Wraps existing SDKs (LeRobot, UR, ABB, etc.)

**Quick Example:**

```python
from inferencekit import InferenceModel
from phyai.robots import SO101

policy = InferenceModel.load("./exports/act_policy")
robot = SO101.from_config("robot.yaml")

with robot:
    policy.reset()
    for _ in range(1000):
        obs = robot.get_observation()
        action = policy.predict(obs)["action"]
        robot.send_action(action)
```

**Wrapper Architecture:**

| Layer                 | Description                           | Examples         |
| --------------------- | ------------------------------------- | ---------------- |
| **Universal wrapper** | Flexible, accepts robot_type + kwargs | `LeRobotRobot`   |
| **Specific wrappers** | IDE autocomplete, explicit params     | `SO101`, `Aloha` |
| **External SDKs**     | Underlying implementations            | `lerobot.robots` |

**For detailed robot interface design, see:** [Robot Interface Design](../robot_interface_design.md)

---

## End-to-End Deployment Patterns

### Minimal Inference (inferencekit Only)

Using only inferencekit (no phyai):

```python
from inferencekit import InferenceModel

# Load exported model
model = InferenceModel.load("./exports/my_model")

# Run inference
inputs = {"image": image_array, "state": state_array}
outputs = model.predict(inputs)
```

### Camera + Inference

Adding phyai for camera support:

```python
from inferencekit import InferenceModel
from phyai.cameras import RealSense

policy = InferenceModel.load("./exports/act_policy")
camera = RealSense(fps=30)

with camera:
    policy.reset()
    while running:
        frame = camera.read()
        observation = {"image": frame, "state": get_state()}
        action = policy.predict(observation)["action"]
```

### Full Robot Control Loop

Complete deployment with camera, inference, and robot:

```python
from inferencekit import InferenceModel
from phyai.cameras import RealSense
from phyai.robots import SO101

# Load components
policy = InferenceModel.load("./exports/act_policy")
camera = RealSense(fps=30)
robot = SO101.from_config("robot.yaml")

# Run control loop
with camera, robot:
    policy.reset()

    while not done:
        image = camera.read()
        state = robot.get_observation()["state"]

        observation = {"images": {"top": image}, "state": state}
        action = policy.predict(observation)["action"]

        robot.send_action(action)
```

### Edge Deployment Considerations

| Consideration         | Recommendation                                      |
| --------------------- | --------------------------------------------------- |
| **Backend selection** | OpenVINO for Intel, ONNX+TensorRT for NVIDIA Jetson |
| **Memory**            | Use FP16/INT8 quantization                          |
| **Latency**           | Warm up model before control loop                   |
| **Reliability**       | Handle inference failures gracefully                |

### Safety Considerations

```python
from phyai.callbacks import ActionSafetyCallback

safety = ActionSafetyCallback(
    action_min=-1.0,
    action_max=1.0,
    velocity_limit=0.1,
)

model = InferenceModel.load(
    "./exports/act_policy",
    callbacks=[safety],
)
```

---

## Compatibility with LeRobot Export

### Shared Design Principles

Both frameworks agree on:

1. **Self-describing packages**: Manifest defines how to load and run
2. **Multi-backend artifacts**: Same package can contain ONNX, OpenVINO, etc.
3. **Pure data manifests**: No code references, just JSON configuration
4. **Minimal runtime dependencies**: Load without importing training code

### PolicyPackage Format

LeRobot defines the `PolicyPackage` format with `manifest.json`. The inference pattern is determined by the structure of the `inference` field:

- `inference: null` → Single-pass (ACT, Groot)
- `inference` has `scheduler` → Iterative (Diffusion)
- `inference` has `encoder_artifact` → Two-phase VLA (PI0, SmolVLA)

```json
{
  "format": "policy_package",
  "version": "1.0",
  "policy": {
    "name": "act_policy",
    "source": { "repo_id": "user/repo", "revision": "main" }
  },
  "artifacts": {
    "onnx": "model.onnx",
    "openvino": "model.xml"
  },
  "io": {
    "inputs": [
      {
        "name": "observation.image",
        "dtype": "float32",
        "shape": [1, 3, 480, 640]
      }
    ],
    "outputs": [{ "name": "action", "dtype": "float32", "shape": [1, 100, 6] }]
  },
  "action": {
    "dim": 6,
    "chunk_size": 100,
    "n_action_steps": 100,
    "representation": "absolute"
  },
  "inference": null,
  "normalization": {
    "type": "min_max",
    "artifact": "stats.safetensors",
    "input_features": ["observation.state"],
    "output_features": ["action"]
  }
}
```

### Runner Mapping

| LeRobot Inference Pattern    | Manifest Structure                   | phyai Runner                                 |
| ---------------------------- | ------------------------------------ | -------------------------------------------- |
| Single-pass (ACT, Groot)     | `inference: null`                    | `ActionChunkingRunner` or `SinglePassRunner` |
| Iterative (Diffusion)        | `inference.scheduler` present        | `IterativeRunner`                            |
| Two-phase VLA (PI0, SmolVLA) | `inference.encoder_artifact` present | `TwoPhaseRunner`                             |

### Runtime Compatibility

| Feature                | LeRobot Runtime | phyai           |
| ---------------------- | --------------- | --------------- |
| Load PolicyPackage     | ✅              | ✅ (via plugin) |
| Single-pass inference  | ✅              | ✅              |
| Iterative inference    | ✅              | ✅              |
| Two-phase VLA          | ✅              | ✅              |
| Action queue           | ✅              | ✅              |
| Normalization          | ✅              | ✅              |
| Callbacks              | ❌              | ✅              |
| Multi-backend fallback | ❌              | ✅              |

### Maintenance Strategy

| Principle                    | Implementation                            |
| ---------------------------- | ----------------------------------------- |
| **Single source of truth**   | LeRobot owns PolicyPackage format         |
| **No circular dependencies** | phyai consumes, never imports LeRobot     |
| **Conformance tests**        | Shared test suite validates compatibility |

**For detailed LeRobot integration design, see:** [LeRobot Integration Design](../inferencekit_lerobot_integration.md)

---

## API Reference

### Robotics Runners

```python
from phyai.runners import (
    IterativeRunner,         # Diffusion denoising
    TwoPhaseRunner,          # VLA with encoder caching (PI0, SmolVLA)
    ActionChunkingRunner,    # Action queue management (ACT, Groot)
)
```

### Robotics Callbacks

```python
from phyai.callbacks import (
    ActionSafetyCallback,    # Action clamping, velocity limiting
    EpisodeLoggingCallback,  # Episode data recording
)
```

### Cameras

```python
from phyai.cameras import (
    Camera,                  # ABC
    Webcam,                  # USB cameras
    RealSense,               # Intel depth cameras
    Basler,                  # Industrial cameras
    VideoFile,               # Video playback
    ImageFolder,             # Image sequences
)
```

### Robots

```python
from phyai.robots import (
    Robot,                   # ABC
    LeRobotRobot,            # Universal LeRobot wrapper
    SO101,                   # SO-101 robot
    Aloha,                   # Aloha robot
)
```

### Extension Points

| Extension     | How to Extend                            |
| ------------- | ---------------------------------------- |
| New runner    | Implement `InferenceRunner`              |
| New callback  | Subclass `Callback`                      |
| New framework | Implement plugin with `@register_format` |
| New camera    | Subclass `Camera`                        |
| New robot     | Subclass `Robot`                         |

---

## Related Documents

- **[Overview](./overview.md)** - High-level architecture of inference framework
- **[inferencekit Design](./inferencekit_design.md)** - Generic inference package
- **[Camera Interface Design](../camera_interface_design.md)** - Detailed camera interface specification
- **[Robot Interface Design](../robot_interface_design.md)** - Detailed robot interface specification
- **[LeRobot Integration](../inferencekit_lerobot_integration.md)** - LeRobot PolicyPackage integration

---

_Document Version: 1.0_
_Last Updated: 2026-01-28_
