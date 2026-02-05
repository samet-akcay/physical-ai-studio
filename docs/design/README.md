# Inference Framework Documentation

This directory contains the design documentation for the Physical AI Inference Framework (**physical‑ai‑framework**), which consists of three complementary layers:

1. **inferencekit** - Domain-agnostic inference framework
2. **getiaction** - End-to-end robotics stack (library + application)
3. **physical‑ai‑framework** - Lightweight deployment shell

> **Codename**: **physical‑ai‑framework** is the codename for the deployment/inference repo.

---

## Quick Navigation

### Start Here

- **[Overview](./overview.md)** - Big-picture architecture and the three-layer model

### Architecture & Strategy

- **[Library-First Pipeline](./library_first_pipeline.md)** - Library-first design for robot, teleop, and data collection APIs
- **[Modular Robot API & Deployment](./modular_robot_api_deployment.md)** - Modular Robot API with optional extras for edge deployment

### Core Design Documents

- **[inferencekit Design](./inferencekit_design.md)** - Generic inference package design

  - InferenceModel, RuntimeAdapter, InferenceRunner
  - Backend support (OpenVINO, ONNX, TensorRT, Torch)
  - Callback system
  - Metadata format

### Component APIs

- **[Teleoperation API](./teleoperation_api.md)** - Leader/follower semantics, lifecycle, safety
- **[Data Collection API](./data_collection_api.md)** - Dataset writer, metadata, HF Hub upload
- **[Deployment Shell](./deployment_shell.md)** - CLI, configuration, and deployment patterns

### Detailed Component Designs

- **[Camera Interface Design](./camera_interface_design.md)** - Detailed camera interface specification
- **[Robot Interface Design](./robot_interface_design.md)** - Detailed robot interface specification
- **[LeRobot Integration](./inferencekit_lerobot_integration.md)** - LeRobot PolicyPackage integration

### Internal Notes

- **[LeRobot Export Suggestions](./lerobot_export_suggestions.md)** - Improvement suggestions for LeRobot export API (internal)

---

## Document Structure

```
docs/design/
├── README.md                           # This file
├── overview.md                         # Big-picture architecture
├── library_first_pipeline.md           # Library-first pipeline design
├── modular_robot_api_deployment.md     # Modular Robot API & deployment
├── inferencekit_design.md              # Generic inference package (Part I)
├── teleoperation_api.md                # Teleoperation API design
├── data_collection_api.md              # Data collection API design
├── deployment_shell.md                 # Deployment shell design
├── camera_interface_design.md          # Detailed camera interface
├── robot_interface_design.md           # Detailed robot interface
├── inferencekit_lerobot_integration.md # LeRobot integration
└── lerobot_export_suggestions.md       # Internal: LeRobot API improvement notes
```

---

## Reading Guide

### For New Readers

1. Start with **[Overview](./overview.md)** to understand the big-picture architecture
2. Read **[inferencekit Design](./inferencekit_design.md)** to understand the generic inference package
3. Explore the component APIs for specific use cases

### For Implementation

1. Read the core design documents (inferencekit)
2. Refer to detailed component designs as needed:
   - [Camera Interface Design](./camera_interface_design.md) for camera implementation
   - [Robot Interface Design](./robot_interface_design.md) for robot implementation
   - [LeRobot Integration](./inferencekit_lerobot_integration.md) for LeRobot support

### For Deployment

- **[Deployment Shell](./deployment_shell.md)** - CLI and configuration patterns for physical‑ai‑framework

### For Integration

- **getiaction users**: See component APIs (Teleoperation, Data Collection)
- **LeRobot users**: See [LeRobot Integration](./inferencekit_lerobot_integration.md)

---

## Architecture Overview

The Physical AI Framework follows a three-layer architecture:

```
                inferencekit
                      ▲
                      │
                  getiaction
                      ▲
                      │
           physical‑ai‑framework (shell)
```

**Layer responsibilities:**

- **inferencekit**: Core inference, backend abstraction, runners, callbacks
- **getiaction**: Robot/camera/teleop APIs, policies, preprocessing, LeRobot compatibility
- **physical‑ai‑framework**: CLI, packaging, deployment docs

---

## Package Naming

### Current Status (2026-02-05)

| Package           | Name                    | Status                               |
| ----------------- | ----------------------- | ------------------------------------ |
| Generic Inference | `inferencekit`          | Proposed replacement for `model_api` |
| Robotics Stack    | `getiaction`            | Current library + application        |
| Deployment Shell  | `physical‑ai‑framework` | Codename for deployment repo         |

_Names are subject to marketing/branding review._

---

## Contributing

When updating these documents:

1. **Keep documents focused**: Each document should cover a single aspect
2. **Maintain cross-references**: Link between related documents
3. **Update version and date**: At the bottom of each document
4. **Update this README**: If you add new documents or change structure

---

## Version History

| Version | Date       | Changes                                          |
| ------- | ---------- | ------------------------------------------------ |
| 2.0     | 2026-02-05 | Restructured with physical‑ai‑framework codename |
| 1.2     | 2026-01-28 | Added modular installation plan                  |
| 1.1     | 2026-01-28 | Added TwoPhaseRunner, LeRobot export suggestions |
| 1.0     | 2026-01-28 | Initial split from combined document             |

---

_Last Updated: 2026-02-05_
