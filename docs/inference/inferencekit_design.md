# inferencekit Design Document

**Domain-agnostic inference framework for unified model loading and prediction across multiple backends.**

---

## Table of Contents

- [Overview](#overview)
  - [Purpose](#purpose)
  - [Relationship to model_api](#relationship-to-model_api)
  - [Design Goals](#design-goals)
  - [Non-Goals](#non-goals)
- [Architecture](#architecture)
  - [Package Structure](#package-structure)
  - [Design Principles](#design-principles)
- [Core Components](#core-components)
  - [InferenceModel](#inferencemodel)
  - [RuntimeAdapter](#runtimeadapter)
  - [InferenceRunner](#inferencerunner)
  - [Callback System](#callback-system)
  - [Preprocessors and Postprocessors](#preprocessors-and-postprocessors)
  - [Metadata Format](#metadata-format)
- [Supported Backends](#supported-backends)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Appendix: Design Rationale](#appendix-design-rationale)

---

## Overview

### Purpose

The **inferencekit** package is a domain-agnostic inference framework that standardizes how we load exported models, run predictions, and swap execution backends. It provides:

- Unified `InferenceModel.load()` + `predict()` API
- Backend abstraction (OpenVINO, ONNX, TensorRT, Torch)
- Callback system for instrumentation
- Metadata-driven configuration

**This package knows nothing about robotics, cameras, or physical hardware.** It is a pure inference toolkit that can be used for any ML domain: computer vision, NLP, anomaly detection, robotics, etc.

### Relationship to model_api

The existing `model_api` package provides model inference interfaces. However:

- The name doesn't emphasize inference and can be interpreted as "interfaces only"
- It lacks a clear extensibility model for different inference patterns
- It only supports OpenVINO

**Proposal:** The `inferencekit` package is designed to replace `model_api` with:

- Clearer naming that emphasizes inference
- Composable architecture (runners, adapters, callbacks)
- Better extensibility for domain-specific needs

| Aspect        | model_api | inferencekit                 |
| ------------- | --------- | ---------------------------- |
| Name clarity  | Ambiguous | Clear inference focus        |
| Extensibility | Limited   | Runners, adapters, callbacks |
| Status        | Existing  | Proposed replacement         |

_Final decision on naming and replacement pending marketing/branding review._

### Design Goals

| Goal                         | Description                                                      |
| ---------------------------- | ---------------------------------------------------------------- |
| **G1: Unified API**          | Single `InferenceModel.load()` + `predict()` across all backends |
| **G2: Backend Agnostic**     | Support OpenVINO, ONNX, TensorRT, Torch without code changes     |
| **G3: Extensible**           | Easy to add new backends, runners, callbacks                     |
| **G4: Minimal Dependencies** | Core has few requirements; optional extras per backend           |
| **G5: Domain Agnostic**      | No robotics, vision, or domain-specific code                     |

### Non-Goals

| Non-Goal                | Rationale                                |
| ----------------------- | ---------------------------------------- |
| Robotics support        | Belongs in Physical AI Framework (phyai) |
| Camera/robot interfaces | Belongs in Physical AI Framework (phyai) |
| Training infrastructure | Separate concern                         |
| Framework-specific code | Use plugins in consuming packages        |

---

## Architecture

### Package Structure

```
inferencekit/                                # Proposed: replaces model_api
├── __init__.py                              # Public API: InferenceModel
├── model.py                                 # InferenceModel - main entry point
├── runners/
│   ├── __init__.py
│   ├── base.py                              # InferenceRunner ABC
│   └── single_pass.py                       # SinglePassRunner (default)
├── adapters/
│   ├── __init__.py                          # get_adapter() factory
│   ├── base.py                              # RuntimeAdapter ABC
│   ├── openvino.py                          # OpenVINO backend
│   ├── onnx.py                              # ONNX Runtime backend
│   ├── tensorrt.py                          # TensorRT backend (future)
│   └── torch_export.py                      # Torch Export IR / ExecuTorch
├── callbacks/
│   ├── __init__.py
│   ├── base.py                              # Callback ABC
│   ├── timing.py                            # TimingCallback
│   └── logging.py                           # LoggingCallback
├── preprocessors/
│   ├── __init__.py
│   └── base.py                              # Preprocessor ABC
├── postprocessors/
│   ├── __init__.py
│   └── base.py                              # Postprocessor ABC
├── io/
│   ├── __init__.py
│   ├── metadata.py                          # Metadata loading (YAML/JSON)
│   └── instantiate.py                       # class_path + init_args
└── plugins/
    ├── __init__.py                          # Plugin registry
    └── base.py                              # Plugin ABC
```

### Design Principles

| Principle                        | Description                                            |
| -------------------------------- | ------------------------------------------------------ |
| **Domain Agnostic**              | No robotics, vision, or domain-specific code           |
| **Backend Agnostic**             | Core interfaces don't depend on specific backends      |
| **Progressive Disclosure**       | Simple API for common cases, full control available    |
| **Composition over Inheritance** | Runners, callbacks, adapters are composable            |
| **Minimal Dependencies**         | Core has few requirements; optional extras per backend |

---

## Core Components

### InferenceModel

The main entry point for inference. Orchestrates runners, adapters, and callbacks.

**Design Philosophy:**

90% of users should only need:

```python
from inferencekit import InferenceModel

model = InferenceModel.load("./exports/my_model")
outputs = model.predict(inputs)
```

Progressive customization for advanced users:

```python
# Tier 2: Override parameters
model = InferenceModel.load(
    "./exports/my_model",
    backend="onnx",
    device="cuda",
)

# Tier 3: Explicit components
from inferencekit.callbacks import TimingCallback

model = InferenceModel.load(
    "./exports/my_model",
    callbacks=[TimingCallback()],
)
```

**API:**

```python
class InferenceModel:
    """Unified inference interface for exported models.

    Automatically detects backend, device, and configuration from
    export directory metadata.
    """

    @classmethod
    def load(
        cls,
        path: str | Path,
        backend: str | None = None,
        device: str = "auto",
        callbacks: list[Callback] | None = None,
        **kwargs,
    ) -> "InferenceModel":
        """Load model from export directory.

        Args:
            path: Directory containing exported model and metadata
            backend: Backend to use (auto-detected if None)
            device: Device for inference ("auto", "cpu", "cuda", "CPU", "GPU")
            callbacks: Optional callbacks for instrumentation
            **kwargs: Additional arguments passed to runner/adapter

        Returns:
            Configured InferenceModel ready for inference
        """
        ...

    def predict(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run inference on inputs.

        Args:
            inputs: Dictionary mapping input names to arrays/tensors

        Returns:
            Dictionary mapping output names to arrays/tensors
        """
        ...

    def reset(self) -> None:
        """Reset model state (for stateful runners)."""
        ...

    def __enter__(self) -> "InferenceModel":
        """Context manager entry."""
        ...

    def __exit__(self, *args) -> None:
        """Context manager exit - cleanup resources."""
        ...
```

### RuntimeAdapter

Adapters execute **one forward pass** on a specific backend. Intentionally stateless.

```python
class RuntimeAdapter(ABC):
    """Abstract base class for backend-specific inference."""

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.config = kwargs

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load model from disk."""
        ...

    @abstractmethod
    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run single forward pass."""
        ...

    @property
    @abstractmethod
    def input_names(self) -> list[str]:
        """Model input names."""
        ...

    @property
    @abstractmethod
    def output_names(self) -> list[str]:
        """Model output names."""
        ...
```

### InferenceRunner

Runners define **how inference runs** - the algorithm, not what happens to outputs.

The generic package provides only `SinglePassRunner`:

```python
class SinglePassRunner(InferenceRunner):
    """Default runner for single forward pass models.

    Covers 90% of use cases: classification, detection,
    segmentation, single-output models, etc.
    """

    def run(self, adapter: RuntimeAdapter, inputs: dict) -> dict:
        return adapter.predict(inputs)

    def reset(self) -> None:
        pass  # No state to reset
```

**Note:** Domain-specific runners (e.g., `IterativeRunner`, `ActionChunkingRunner`) belong in the Physical AI Framework (phyai), not here.

### Callback System

Lightning-compatible hooks for cross-cutting concerns:

```python
class Callback:
    """Base callback class. Override hooks as needed."""

    def on_predict_start(self, inputs: dict) -> dict | None:
        """Called before prediction. Can modify inputs."""
        pass

    def on_predict_end(self, outputs: dict) -> dict | None:
        """Called after prediction. Can modify outputs."""
        pass

    def on_reset(self) -> None:
        """Called when model state is reset."""
        pass
```

**Built-in callbacks (generic package):**

| Callback          | Purpose               |
| ----------------- | --------------------- |
| `TimingCallback`  | Performance profiling |
| `LoggingCallback` | Prediction logging    |

### Preprocessors and Postprocessors

Transform inputs before inference and outputs after:

```python
class Preprocessor(ABC):
    """Transform inputs before inference."""

    @abstractmethod
    def __call__(self, inputs: dict) -> dict:
        ...

class Postprocessor(ABC):
    """Transform outputs after inference."""

    @abstractmethod
    def __call__(self, outputs: dict) -> dict:
        ...
```

### Metadata Format

Following `jsonargparse` conventions with `class_path` + `init_args`:

```yaml
# metadata.yaml
backend: onnx

runner:
  class_path: inferencekit.runners.SinglePassRunner
  init_args: {}

adapter:
  class_path: inferencekit.adapters.ONNXAdapter
  init_args:
    providers:
      - CUDAExecutionProvider
      - CPUExecutionProvider
```

**Loading priority:**

1. `metadata.yaml`
2. `metadata.yml`
3. `metadata.json`

---

## Supported Backends

| Backend             | Hardware             | Status         | Installation                  |
| ------------------- | -------------------- | -------------- | ----------------------------- |
| **OpenVINO**        | Intel CPU/GPU/NPU    | ✅ Implemented | `pip install openvino`        |
| **ONNX Runtime**    | Cross-platform, CUDA | ✅ Implemented | `pip install onnxruntime-gpu` |
| **TensorRT**        | NVIDIA GPU           | 🔄 Planned     | `pip install tensorrt`        |
| **Torch Export IR** | CPU/CUDA, ExecuTorch | ✅ Implemented | Built-in                      |

---

## Usage Examples

### Basic usage

```python
from inferencekit import InferenceModel

# Load model (auto-detects backend)
model = InferenceModel.load("./exports/my_model")

# Run inference
inputs = {"input": image_array}
outputs = model.predict(inputs)
```

### With explicit backend

```python
model = InferenceModel.load(
    "./exports/my_model",
    backend="openvino",
    device="CPU",
)
```

### With callbacks

```python
from inferencekit.callbacks import TimingCallback, LoggingCallback

model = InferenceModel.load(
    "./exports/my_model",
    callbacks=[TimingCallback(), LoggingCallback()],
)

# Callbacks fire automatically
outputs = model.predict(inputs)
```

### Context manager for resource cleanup

```python
with InferenceModel.load("./exports/my_model") as model:
    outputs = model.predict(inputs)
# Resources automatically cleaned up
```

---

## API Reference

### Main Entry Point

```python
# Main entry point
from inferencekit import InferenceModel

# Load and run inference
model = InferenceModel.load("./exports/my_model")
outputs = model.predict(inputs)
```

### Runners (Generic Package)

```python
from inferencekit.runners import (
    InferenceRunner,      # ABC
    SinglePassRunner,     # Default - covers 90% of models
)
```

### Adapters

```python
from inferencekit.adapters import (
    RuntimeAdapter,       # ABC
    OpenVINOAdapter,      # Intel devices
    ONNXAdapter,          # Cross-platform
    TorchExportAdapter,   # PyTorch
    get_adapter,          # Factory function
)
```

### Callbacks

```python
from inferencekit.callbacks import (
    Callback,             # ABC
    TimingCallback,       # Performance profiling
    LoggingCallback,      # Prediction logging
)
```

### Extension Points

| Extension      | How to Extend               |
| -------------- | --------------------------- |
| New backend    | Implement `RuntimeAdapter`  |
| New runner     | Implement `InferenceRunner` |
| New callback   | Subclass `Callback`         |
| Preprocessing  | Implement `Preprocessor`    |
| Postprocessing | Implement `Postprocessor`   |

---

## Appendix: Design Rationale

### Why a separate inference package?

1. **Reusability**: Same core can be used across robotics (phyai), computer vision (Geti), NLP, etc.
2. **Clear boundaries**: Generic concerns (backends, metadata) separated from domain concerns (cameras, robots)
3. **Easier testing**: Domain-agnostic package has fewer dependencies
4. **Progressive enhancement**: Users can use just inferencekit or add phyai for robotics

### Why runners are separate from adapters?

- **Adapters** handle backend-specific execution (ONNX vs OpenVINO)
- **Runners** handle algorithm-specific patterns (single-pass vs iterative)
- This separation allows N backends × M inference patterns without N×M implementations

### Why callbacks instead of inheritance?

- **Composability**: Mix and match multiple behaviors (timing + logging + safety)
- **Reusability**: Same callback works across all models
- **Maintainability**: Add new cross-cutting concerns without changing core code
- **Familiarity**: Lightning users already understand this pattern

### Why metadata-driven configuration?

- **Self-describing exports**: Model knows how to load itself
- **Version compatibility**: Can evolve format over time
- **Override flexibility**: Users can override any parameter at load time
- **Multi-framework support**: Each framework can provide its own metadata

---

## Related Documents

- **[Overview](./overview.md)** - High-level architecture of inference framework
- **[phyai Design](./phyai_design.md)** - Physical AI Framework built on inferencekit
- **[Universal Inference Package Design](../inference_package_design.md)** - Original detailed design

---

_Document Version: 1.0_
_Last Updated: 2026-01-28_
