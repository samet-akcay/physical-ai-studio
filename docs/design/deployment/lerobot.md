# physical‑ai‑framework: LeRobot Integration Design

**Status**: Proposal
**Author**: [Your Name]
**Date**: 2026-01-13
**Relates to**: [LeRobot Policy Export Design](./policy_export_design.md)

> **Important: LeRobot export is our proposal, not an agreed standard.**
> The PolicyPackage format (`manifest.json`) described in this document is a design we have proposed to the LeRobot team. It has **not yet been reviewed or accepted** upstream. If the LeRobot team adopts a different export format or modifies the proposed schema, this integration design will need to adapt accordingly. The architectural approach (built‑in format loader, no lerobot dependency at runtime) remains valid regardless of the final format — only the loader implementation would change.

---

## Executive Summary

This document describes how **physical‑ai‑framework** would integrate with LeRobot's proposed PolicyPackage format. The integration would be implemented as a **built‑in format loader** — the framework reads `manifest.json` (pure JSON, no lerobot import) and maps `policy.kind` to built‑in runners. No LeRobot dependency would be needed at deployment time.

**Key principle**: LeRobot would define the package format; physical‑ai‑framework consumes it via a built‑in format loader. No circular dependencies. No external plugin needed.

**Note on status**: The PolicyPackage export format is our proposal to the LeRobot team (see [LeRobot Export Suggestions](../internal/lerobot-export-suggestions.md)). The format details below reflect our proposed design. The integration approach is sound regardless of the final format the LeRobot team adopts.

---

## 1. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    physical‑ai‑framework                        │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Adapters   │  │  Built‑in    │  │     Callbacks        │  │
│  │  (backends)  │  │   Runners    │  │  (instrumentation)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │               Built‑in Format Loaders                    │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │ manifest.json│  │metadata.yaml│  │  Custom Format  │   │  │
│  │  │  (LeRobot)  │  │ (getiaction)│  │  (external)     │   │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ reads (pure file I/O)
                              ▼
                    ┌──────────────────┐
                    │  PolicyPackage   │
                    │  (LeRobot format)│
                    │                  │
                    │  manifest.json   │
                    │  artifacts/      │
                    └──────────────────┘
```

---

## 2. Shared Contract (Proposed)

physical‑ai‑framework and LeRobot would agree on the **PolicyPackage** format defined in the [LeRobot Policy Export Design](./policy_export_design.md). This format is our proposal — the final contract depends on LeRobot team acceptance.

### Package Detection

A directory is a LeRobot PolicyPackage if:

1. It contains `manifest.json`
2. The manifest has `"format": "policy_package"`

```python
def is_lerobot_package(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text())
    return manifest.get("format") == "policy_package"
```

### Manifest Fields Used

| Field           | physical‑ai‑framework Usage                                                            |
| --------------- | -------------------------------------------------------------------------------------- |
| `format`        | Package type detection                                                                 |
| `version`       | Schema compatibility check                                                             |
| `policy.kind`   | Runner selection (`single_shot` → `SinglePassRunner`, `iterative` → `IterativeRunner`) |
| `artifacts`     | Backend artifact paths                                                                 |
| `io`            | Input/output validation                                                                |
| `action`        | Action semantics (chunk_size, n_action_steps)                                          |
| `iterative`     | Loop parameters (num_steps, scheduler)                                                 |
| `normalization` | Normalizer configuration                                                               |
| `x-physical-ai` | Extension fields (callbacks, adapter options)                                          |

---

## 3. Format Loader Implementation

### Registration

```python
# physical_ai/format_loaders/lerobot.py

from physical_ai.format_loaders import register_format

@register_format("policy_package")
class LeRobotFormatLoader:
    """Built‑in format loader for LeRobot PolicyPackages."""

    @staticmethod
    def detect(path: Path) -> bool:
        """Check if path is a LeRobot PolicyPackage."""
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("format") == "policy_package"
        except (json.JSONDecodeError, KeyError):
            return False

    @staticmethod
    def load(
        path: Path,
        backend: str | None = None,
        device: str = "cpu",
        **kwargs
    ) -> "InferenceModel":
        """Load a PolicyPackage into an InferenceModel."""
        manifest = json.loads((path / "manifest.json").read_text())

        # Validate schema version
        version = manifest.get("version", "1.0")
        if not version.startswith("1."):
            raise ValueError(f"Unsupported manifest version: {version}")

        # Select backend
        backend = backend or _select_default_backend(manifest)
        artifact_path = path / manifest["artifacts"][backend]

        # Create adapter (via inference core)
        adapter = get_adapter(backend)(artifact_path, device=device)

        # Select runner based on policy kind
        kind = manifest["policy"]["kind"]
        runner = _create_runner(kind, manifest, **kwargs)

        # Create normalizer
        normalizer = _create_normalizer(path, manifest)

        # Load callbacks from extensions
        callbacks = _load_callbacks(manifest)

        return InferenceModel(
            adapter=adapter,
            runner=runner,
            normalizer=normalizer,
            callbacks=callbacks,
            metadata=manifest,
        )


def _create_runner(kind: str, manifest: dict, **kwargs) -> InferenceRunner:
    """Map LeRobot policy kind to physical‑ai runner."""
    if kind == "single_shot":
        return SinglePassRunner()

    elif kind == "iterative":
        iter_config = manifest.get("iterative", {})
        return IterativeRunner(
            num_steps=kwargs.get("num_steps", iter_config.get("num_steps", 10)),
            scheduler=kwargs.get("scheduler", iter_config.get("scheduler", "euler")),
            timestep_spacing=iter_config.get("timestep_spacing", "linear"),
        )

    else:
        raise ValueError(f"Unknown policy kind: {kind}")


def _load_callbacks(manifest: dict) -> list[Callback]:
    """Load callbacks from x-physical-ai extension."""
    callbacks = []
    ext = manifest.get("x-physical-ai", {})

    for cb_config in ext.get("callbacks", []):
        if isinstance(cb_config, str):
            # Short form: "timing" -> TimingCallback()
            callbacks.append(get_callback(cb_config)())
        elif isinstance(cb_config, dict):
            # Full form: {"class_path": "...", "init_args": {...}}
            callbacks.append(instantiate(cb_config))

    return callbacks
```

### Installation

The LeRobot format loader is **built‑in** — it ships with physical‑ai‑framework. No extra install needed.

```bash
# This is all you need to run LeRobot-exported models
pip install physical-ai-framework
```

The format loader reads `manifest.json` (pure JSON parsing) and maps `policy.kind` to built‑in runners. No `lerobot` import. No `physical-ai-framework[lerobot]` extra.

---

## 4. Usage Examples

### Basic Usage (Unified API)

```python
from physical_ai import InferenceModel

# Load LeRobot package (auto-detected via plugin)
model = InferenceModel("./pi0_exported")

# Run inference (raw outputs)
observation = {
    "observation.images.top": image_array,
    "observation.state": state_array,
}
outputs = model(observation)
action_chunk = outputs["action"]
```

### With Callbacks

```python
from physical_ai import InferenceModel
from physical_ai.callbacks import TimingCallback, LoggingCallback

model = InferenceModel(
    "./pi0_exported",
    callbacks=[
        TimingCallback(),
        LoggingCallback(log_inputs=False, log_outputs=True),
    ],
)

# Callbacks fire automatically on predict
action = model(observation)
# -> logs timing and output summary
```

### Override Runner Parameters

```python
# Override num_steps at load time (no re-export needed)
model = InferenceModel(
    "./pi0_exported",
    num_steps=20,  # Override manifest default of 10
    scheduler="ddim",
)
```

### Real-Time Control (Policy API)

```python
from physical_ai import InferenceModel

policy = InferenceModel("./pi0_exported")
policy.reset()

while not done:
    action = policy.select_action(observation)
    observation, reward, done, info = env.step(action)
```

### Explicit Backend Selection

```python
# Use specific backend
model = InferenceModel(
    "./pi0_exported",
    backend="onnx",
    device="cuda:0",
)

# Or with adapter options
model = InferenceModel(
    "./pi0_exported",
    backend="onnx",
    adapter_options={
        "providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    },
)
```

---

## 5. Extension Fields

physical‑ai‑framework-specific configuration can be embedded in the manifest under `x-physical-ai`:

```json
{
  "format": "policy_package",
  "version": "1.0",

  "policy": { ... },
  "artifacts": { ... },

"x-physical-ai": {
    "callbacks": [
      "timing",
      {"class_path": "myproject.callbacks.SafetyCallback", "init_args": {"max_velocity": 1.0}}
    ],
    "adapter": {
      "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
      "graph_optimization_level": "all"
    },
    "preprocessors": [
{"class_path": "physical_ai.preprocessors.ImageNormalize", "init_args": {"mean": [0.485, 0.456, 0.406]}}
    ]
  }
}
```

### Extension Schema

| Field            | Type                          | Description             |
| ---------------- | ----------------------------- | ----------------------- |
| `callbacks`      | `list[str \| CallbackConfig]` | Callbacks to attach     |
| `adapter`        | `dict`                        | Adapter/backend options |
| `preprocessors`  | `list[PreprocessorConfig]`    | Input preprocessors     |
| `postprocessors` | `list[PostprocessorConfig]`   | Output postprocessors   |

**Note**: LeRobot ignores `x-physical-ai` fields entirely. They are only read by physical‑ai‑framework.

---

## 6. Runner Mapping

### LeRobot `policy.kind` → physical‑ai Runner

| LeRobot Kind  | physical‑ai Runner | Notes                            |
| ------------- | ------------------ | -------------------------------- |
| `single_shot` | `SinglePassRunner` | Direct forward pass              |
| `iterative`   | `IterativeRunner`  | Configurable loop with scheduler |

### IterativeRunner Configuration

```python
class IterativeRunner(InferenceRunner):
    """Runner for iterative/flow-matching policies."""

    def __init__(
        self,
        num_steps: int = 10,
        scheduler: str = "euler",
        timestep_spacing: str = "linear",
        timestep_range: tuple[float, float] = (1.0, 0.0),
    ):
        self.num_steps = num_steps
        self.scheduler = scheduler
        self.timestep_spacing = timestep_spacing
        self.timestep_range = timestep_range

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        # Initialize from noise
        action_shape = self._infer_action_shape(inputs)
        x_t = np.random.randn(*action_shape).astype(np.float32)

        # Generate timesteps
        timesteps = self._generate_timesteps()
        dt = -1.0 / self.num_steps

        # Iterative denoising
        for t in timesteps:
            step_inputs = {
                **inputs,
                "x_t": x_t,
                "timestep": np.array([t], dtype=np.float32),
            }
            v_t = adapter.predict(step_inputs)["v_t"]
            x_t = self._step(x_t, v_t, dt)

        return {"action": x_t}

    def _step(self, x: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        if self.scheduler == "euler":
            return x + dt * v
        elif self.scheduler == "ddim":
            # DDIM update rule
            ...
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
```

---

## 7. Callbacks for Robotics

physical‑ai‑framework provides callbacks useful for robotics applications:

### ActionSafetyCallback

```python
# physical_ai/callbacks/safety.py

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

    def on_reset(self):
        self._last_action = None
```

### EpisodeLoggingCallback

```python
# physical_ai/callbacks/logging.py

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

    def on_reset(self):
        if self._episode_data:
            self._save_episode()
        self._episode_data = []
        self._episode_count += 1

    def _save_episode(self):
        path = self.log_dir / f"episode_{self._episode_count:04d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._episode_data))
```

---

## 8. Metadata File Format

physical‑ai‑framework supports both YAML and JSON metadata:

### Loading Priority

```python
def load_metadata(path: Path) -> dict:
    """Load metadata from package directory."""
    # Priority order
    for name in ["metadata.yaml", "metadata.yml", "manifest.json"]:
        file_path = path / name
        if file_path.exists():
            return _parse_file(file_path)
    raise FileNotFoundError(f"No metadata found in {path}")
```

### Format Comparison

| Aspect               | `manifest.json` (LeRobot) | `metadata.yaml` (physical‑ai) |
| -------------------- | ------------------------- | ----------------------------- |
| Primary use          | LeRobot packages          | physical‑ai native packages   |
| Human editing        | Harder (no comments)      | Easier (comments, cleaner)    |
| Parsing speed        | Faster                    | Slightly slower               |
| `class_path` support | No (pure data)            | Yes (power users)             |

### Unified Loading

physical‑ai‑framework handles both transparently:

```python
# Works with LeRobot packages (manifest.json)
model = InferenceModel("./lerobot_package")

# Works with physical‑ai packages (metadata.yaml)
model = InferenceModel("./physical_ai_package")

# Format detected automatically
```

---

## 9. Testing Compatibility

### Conformance Test Suite

```python
# tests/format_loaders/test_lerobot_loader.py

class TestLeRobotFormatLoaderConformance:
"""Verify physical‑ai‑framework correctly loads LeRobot packages."""

    def test_detect_lerobot_package(self, lerobot_package_path):
        """Format loader detects LeRobot packages."""
        assert LeRobotFormatLoader.detect(lerobot_package_path)

    def test_load_single_shot(self, act_package_path):
        """Load single_shot policy."""
        model = InferenceModel(act_package_path)
        assert isinstance(model.runner, SinglePassRunner)

    def test_load_iterative(self, pi0_package_path):
        """Load iterative policy."""
        model = InferenceModel(pi0_package_path)
        assert isinstance(model.runner, IterativeRunner)
        assert model.runner.num_steps == 10  # from manifest

    def test_override_num_steps(self, pi0_package_path):
        """Override iterative params at load time."""
        model = InferenceModel(pi0_package_path, num_steps=20)
        assert model.runner.num_steps == 20

    def test_parity_with_lerobot_runtime(self, pi0_package_path):
        """Output matches LeRobot's own runtime."""
# Load with physical‑ai‑framework
        ik_model = InferenceModel(pi0_package_path)

        # Load with LeRobot
        from lerobot.export import load as lerobot_load
        lr_runtime = lerobot_load(pi0_package_path)

        # Compare outputs
        obs = generate_test_observation()
        np.random.seed(42)
        ik_output = ik_model(obs)
        np.random.seed(42)
        lr_output = lr_runtime.predict_action_chunk(obs)

        np.testing.assert_allclose(ik_output["action"], lr_output, rtol=1e-5)
```

---

## 10. Summary

### What physical‑ai‑framework Adds Over LeRobot Runtime

| Feature                             | LeRobot Runtime | physical‑ai‑framework       |
| ----------------------------------- | --------------- | --------------------------- |
| Load PolicyPackage                  | ✓               | ✓                           |
| Single-shot inference               | ✓               | ✓                           |
| Iterative inference                 | ✓               | ✓                           |
| Action queue wrapper                | ✓               | ✓                           |
| Callbacks (timing, logging, safety) | ✗               | ✓                           |
| Multi-backend with fallback         | ✗               | ✓                           |
| Preprocessor/postprocessor chains   | ✗               | ✓                           |
| Plugin system for other formats     | ✗               | ✓ (built‑in format loaders) |
| YAML metadata support               | ✗               | ✓                           |

### Dependency Direction

```
LeRobot ──────────────────────────────────────────────────┐
    │                                                      │
    │ defines                                              │
    ▼                                                      │
PolicyPackage format (manifest.json)                       │
    │                                                      │
    │ consumed by                                          │
    ▼                                                      │
physical‑ai‑framework (built‑in format loader) ◄──────────────────────┘
                               no dependency on LeRobot code
```

**LeRobot does not depend on physical‑ai‑framework.**
**physical‑ai‑framework can load LeRobot packages without importing LeRobot.**

> **Reminder:** This integration depends on LeRobot adopting the proposed PolicyPackage export format. If LeRobot adopts a different format, the format loader implementation changes but the architecture (built‑in loader, no runtime dependency) remains the same.

---

## Related Documents

- **[Strategy](../strategy.md)** - Big-picture architecture
- **[Inference Core Design](./inferencekit.md)** - Domain-agnostic inference layer
- **[LeRobot Export Suggestions](../internal/lerobot-export-suggestions.md)** - Our proposed improvements to LeRobot's export API

---

_Document version: 2.1_
_Last updated: 2026-02-12_
