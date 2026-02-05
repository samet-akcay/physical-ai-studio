# Physical AI Inference Framework - Overview

This document provides a high-level overview of the inference framework architecture. For detailed technical specifications, see the linked design documents.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture Overview](#architecture-overview)
- [Package Descriptions](#package-descriptions)
- [Implementation Roadmap](#implementation-roadmap)
- [Package Naming](#package-naming)
- [Related Documents](#related-documents)

---

## Executive Summary

This project proposes two complementary products for ML inference deployment:

### inferencekit (Generic Inference Package)

- Domain-agnostic inference framework
- Unified API across backends (OpenVINO, ONNX, TensorRT, Torch)
- Proposed replacement for existing `model_api` package
- No knowledge of robotics, cameras, or physical hardware

**Key Features:**

- Single `InferenceModel` API
- Backend abstraction (OpenVINO, ONNX, TensorRT, Torch)
- Callback system for instrumentation
- Metadata-driven configuration

### phyai (Physical AI Framework)

- First open-source, edge-focused physical AI inference framework
- Built on top of inferencekit
- Adds: camera interface, robot interface, robotics-specific runners
- Multi-framework support: geti-action, LeRobot, extensible to others

**Key Features:**

- Camera interface for visual observations
- Robot interface for action execution
- Robotics-specific inference runners (iterative, action chunking)
- Safety callbacks for robot deployment

---

## Architecture Overview

### Layered Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              phyai                                          │
│                     (Physical AI Framework)                                 │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────────────────┐    │
│  │   Camera    │  │   Robot     │  │   Robotics Components             │    │
│  │  Interface  │  │  Interface  │  │   - IterativeRunner               │    │
│  │             │  │             │  │   - ActionChunkingRunner          │    │
│  └─────────────┘  └─────────────┘  │   - ActionSafetyCallback          │    │
│                                    └───────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Framework Integrations                            │    │
│  │   geti-action Plugin  │  LeRobot Plugin  │  Future Frameworks       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ depends on
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          inferencekit                                       │
│                   (Generic Inference Package)                               │
│                  (proposed replacement for model_api)                       │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐     │
│  │ Inference   │  │  Runtime    │  │  Single     │  │    Callback     │     │
│  │   Model     │  │  Adapters   │  │  PassRunner │  │     System      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘     │
│                                                                             │
│  Backends: OpenVINO │ ONNX Runtime │ TensorRT │ Torch Export IR             │
│                                                                             │
│  NO robotics │ NO cameras │ NO robots │ Domain-agnostic                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Relationships

| Component                   | Package      | Purpose                                 |
| --------------------------- | ------------ | --------------------------------------- |
| `InferenceModel`            | inferencekit | Main inference orchestrator             |
| `RuntimeAdapter`            | inferencekit | Backend-specific execution              |
| `SinglePassRunner`          | inferencekit | Default inference pattern               |
| `TimingCallback`            | inferencekit | Generic instrumentation                 |
| `IterativeRunner`           | phyai        | Diffusion/flow matching                 |
| `ActionChunkingRunner`      | phyai        | Temporal action policies                |
| `ActionSafetyCallback`      | phyai        | Robot safety constraints                |
| Camera interface            | phyai        | Visual observation acquisition          |
| Robot interface             | phyai        | Robot control and state reading         |
| geti-action/LeRobot plugins | phyai        | Framework-specific metadata integration |

---

## Package Descriptions

### inferencekit

**Purpose:** Domain-agnostic inference framework that standardizes model loading, prediction, and backend execution.

**Design Goals:**

| Goal                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| **G1: Unified API**          | Single `InferenceModel` across all backends                  |
| **G2: Backend Agnostic**     | Support OpenVINO, ONNX, TensorRT, Torch without code changes |
| **G3: Extensible**           | Easy to add new backends, runners, callbacks                 |
| **G4: Minimal Dependencies** | Core has few requirements; optional extras per backend       |
| **G5: Domain Agnostic**      | No robotics, vision, or domain-specific code                 |

**Non-Goals:**

- Robotics support → belongs in phyai
- Camera/robot interfaces → belongs in phyai
- Training infrastructure → separate concern
- Framework-specific code → use plugins in consuming packages

**Usage Example:**

```python
from inferencekit import InferenceModel

# Auto-detect backend and configuration
model = InferenceModel.load("./exports/my_model")
outputs = model(inputs)
```

**See:** [inferencekit Design Document](./inferencekit_design.md)

### phyai

**Purpose:** Robotics-focused deployment framework adding camera interfaces, robot interfaces, and robotics-specific components on top of inferencekit.

**Design Goals:**

| Goal                         | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| **G1: Multi-Framework**      | Native integration with geti-action, LeRobot, extensible to others |
| **G2: Hardware Abstraction** | Camera and robot interfaces across vendors                         |
| **G3: Edge Optimization**    | Designed for resource-constrained devices                          |
| **G4: Production Safety**    | Action clamping, logging, emergency stops                          |
| **G5: Minimal Overhead**     | Thin layer over generic inference package                          |

**Usage Example:**

```python
from inferencekit import InferenceModel
from phyai.cameras import RealSense
from phyai.robots import SO101

# Load components
policy = InferenceModel.load("./exports/act_policy")
camera = RealSense(...)
robot = SO101.from_config("robot.yaml")

# Run control loop
with camera, robot:
    policy.reset()
    while not done:
        image = camera.read()
        state = robot.get_observation()["state"]
        observation = {"images": {"top": image}, "state": state}
        action = policy(observation)["action"]
        robot.send_action(action)
```

**See:** [phyai Design Document](./phyai_design.md)

---

## Implementation Roadmap

### Phase 1: inferencekit Core (Foundation)

**Scope:** Foundation for all inference

- [ ] `InferenceModel` with `load()` and `predict()`
- [ ] `RuntimeAdapter` ABC and implementations (OpenVINO, ONNX, Torch)
- [ ] `SinglePassRunner` (default)
- [ ] Metadata loading (YAML/JSON)
- [ ] Basic callbacks (`TimingCallback`, `LoggingCallback`)
- [ ] Plugin system for format detection

**Deliverable:** Standalone generic inference package (potential `model_api` replacement)

### Phase 2: phyai - Runners (Robotics Inference)

**Scope:** Robotics-specific inference patterns

- [ ] `IterativeRunner` for diffusion/flow matching
- [ ] `ActionChunkingRunner` for temporal action policies
- [ ] `ActionSafetyCallback`
- [ ] `EpisodeLoggingCallback`

**Deliverable:** Support for all geti-action policy types

### Phase 3: phyai - LeRobot Integration (Interoperability)

**Scope:** Plugin for LeRobot PolicyPackage format

- [ ] `LeRobotPlugin` with `detect()` and `load()`
- [ ] Runner mapping (`single_shot` → `SinglePassRunner`, etc.)
- [ ] Conformance test suite

**Deliverable:** Seamless LeRobot policy deployment

### Phase 4: phyai - Camera Interface (Visual Observations)

**Scope:** Unified camera abstraction

- [ ] `Camera` ABC with core interface
- [ ] Reference-counted sharing
- [ ] Core cameras: `Webcam`, `RealSense`, `VideoFile`
- [ ] Callback system and capability mixins

**Deliverable:** Camera interface as subpackage

### Phase 5: phyai - Robot Interface (Action Execution)

**Scope:** Robot abstraction (pending design finalization)

- [ ] Finalize `Robot` ABC based on team feedback
- [ ] LeRobot robot wrappers
- [ ] Safety interface design

**Deliverable:** Robot interface with LeRobot support

### Phase 6: Production Hardening (Deployment Ready)

**Scope:** Ready for real-world deployment

- [ ] TensorRT adapter
- [ ] Performance optimization
- [ ] Extract camera interface to standalone package

**Deliverable:** Production-ready framework

---

## Package Naming

### inferencekit (Generic Inference Package)

| Name           | Status       | Notes                                             |
| -------------- | ------------ | ------------------------------------------------- |
| `inferencekit` | **Proposed** | Clear purpose, "kit" implies composable parts     |
| `model_api`    | Existing     | Current name, less descriptive of inference focus |

**Proposal:** `inferencekit` replaces `model_api`. Decision pending marketing/branding review.

**Rationale:**

- Clearer naming that emphasizes inference
- Composable architecture (runners, adapters, callbacks)
- Better extensibility for domain-specific needs

### phyai (Physical AI Framework)

| Name    | Status       | Notes                                       |
| ------- | ------------ | ------------------------------------------- |
| `phyai` | **Proposed** | Short, memorable; PyPI availability pending |

**Decision pending:** PyPI resolution and marketing review.

### Camera Interface

| Name       | Status  | Notes                    |
| ---------- | ------- | ------------------------ |
| TBD        | Pending |                          |
| Subpackage | Initial | Start as `phyai.cameras` |

**Will be named when extracted to standalone package.**

---

## Related Documents

### Design Documents

- **[inferencekit Design](./inferencekit_design.md)** - Generic inference package (Part I)
- **[phyai Design](./phyai_design.md)** - Physical AI framework (Part II)
- **[Camera Interface Design](./camera_interface_design.md)** - Detailed camera interface specification
- **[Robot Interface Design](./robot_interface_design.md)** - Detailed robot interface specification

### External References

1. [LeRobot Repository](https://github.com/huggingface/lerobot)
2. [OpenVINO Documentation](https://docs.openvino.ai/)
3. [ONNX Runtime](https://onnxruntime.ai/)

---

## Comparison with Existing Frameworks

| Feature               | LeRobot         | OpenPI            | Isaac GR00T   | This Solution               |
| --------------------- | --------------- | ----------------- | ------------- | --------------------------- |
| **Focus**             | Training + Data | Foundation Models | Humanoids     | Inference Deployment        |
| **Generic inference** | No              | No                | No            | ✅ (separate package)       |
| **Multi-backend**     | Limited         | No                | TensorRT only | ✅ OpenVINO, ONNX, TensorRT |
| **Camera interface**  | Basic           | No                | ROS           | ✅ Multi-vendor             |
| **Robot interface**   | ✅              | Assumes existing  | ROS/Isaac     | ✅ Multi-vendor             |
| **Edge optimization** | Limited         | No                | Jetson        | ✅ Intel, NVIDIA, ARM       |
| **Callback system**   | No              | No                | No            | ✅ Yes                      |

---

_Document Version: 1.0_
_Last Updated: 2026-01-28_
