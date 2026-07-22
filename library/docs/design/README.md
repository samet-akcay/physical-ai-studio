# Library Design Documentation

Architecture and implementation designs for the `physicalai-train` library — the training-side package for Physical AI Studio.

## Module Designs

| Module | Entry Point | Description |
| --- | --- | --- |
| [CLI](cli/overview.md) | [Overview](cli/overview.md) | Command-line interface using PyTorch Lightning CLI |
| [Config](config/overview.md) | [Overview](config/overview.md) | Configuration system (dataclasses, Pydantic, YAML) |
| [Data](data/overview.md) | [Overview](data/overview.md) | Dataset management and data loading |
| [Gyms](gyms/overview.md) | [Overview](gyms/overview.md) | Simulation environments for training |
| [Policy](policy/overview.md) | [Overview](policy/overview.md) | Policy implementations and base classes |
| [Trainer](trainer/overview.md) | [Overview](trainer/overview.md) | Training infrastructure and metrics |
| [Export](export/overview.md) | [Overview](export/overview.md) | Model export (OpenVINO, ONNX, Torch Export) |
| [Inference](inference/overview.md) | [Overview](inference/overview.md) | Production deployment inference |
| [Evaluation](eval/rollout_metric.md) | [Rollout Metric](eval/rollout_metric.md) | Rollout evaluation metrics |
| [Execution](execution/phases.md) | [Phases](execution/phases.md) | Execution phases |

## Component Interface Designs

Runtime-side abstractions defined by the library. See [`components/`](components/) for individual docs.

| Component | Document | Description |
| --- | --- | --- |
| Robot Interface | [Robot Interface](components/robot-interface.md) | Robot ABC, leader/follower wrappers, SDK integration |
| Camera Interface | [Camera Interface](components/camera-interface.md) | `physicalai.capture` camera classes and sharing |
| Benchmarking | [Benchmarking API](components/benchmarking.md) | NumPy-only benchmark protocols, runner, latency metrics |
| Teleoperation | [Teleoperation API](components/teleoperation.md) | Leader/follower semantics, session lifecycle, safety |
| Data Collection | [Data Collection API](components/data-collection.md) | DatasetWriter, episode management, HF Hub upload |

## Model Implementation Guidelines

Per-model process for evaluating, integrating, validating, exporting, and maintaining robot-learning policies. See [`model-guidelines/`](model-guidelines/).

| Document | Description |
| --- | --- |
| [Model Implementation Guidelines](model-guidelines/model-implementation-guidelines.md) | Per-model process for Studio training and Runtime deployment |
| [Reference](model-guidelines/model-implementation-guidelines-reference.md) | Reference details for the guidelines |
| [Intel Hardware Enablement](model-guidelines/intel-enablement-strategy.md) | Cross-team platform & upstream enablement work |
| [Slides](model-guidelines/model-implementation-guidelines-slides.md) | Slide deck companion |

## Architecture

```mermaid
graph TD
    A["physicalai-train"] --> B["cli/"]
    A --> C["config/"]
    A --> D["data/"]
    A --> E["gyms/"]
    A --> F["policy/"]
    A --> G["trainer/"]
    A --> H["export/"]
    A --> I["inference/"]
    A --> J["eval/"]
```

Cross-cutting strategy, deployment, and team-plan designs live in [`docs/design/`](../../../docs/design/README.md).