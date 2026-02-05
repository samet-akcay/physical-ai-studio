# Inference Framework Documentation

This directory contains the design documentation for the Physical AI Inference Framework, which consists of two complementary packages:

1. **inferencekit** - Domain-agnostic inference framework
2. **phyai** - Physical AI framework built on top of inferencekit

---

## Quick Navigation

### Start Here

- **[Overview](./overview.md)** - High-level architecture, executive summary, and implementation roadmap

### Core Design Documents

- **[inferencekit Design](./inferencekit_design.md)** - Generic inference package design

  - InferenceModel, RuntimeAdapter, InferenceRunner
  - Backend support (OpenVINO, ONNX, TensorRT, Torch)
  - Callback system
  - Metadata format

- **[phyai Design](./phyai_design.md)** - Physical AI framework design
  - Robotics-specific runners (IterativeRunner, TwoPhaseRunner, ActionChunkingRunner)
  - Robotics-specific callbacks (ActionSafetyCallback, EpisodeLoggingCallback)
  - Multi-framework support (geti-action, LeRobot)
  - End-to-end deployment patterns

### Internal Notes

- **[LeRobot Export Suggestions](./lerobot_export_suggestions.md)** - Improvement suggestions for LeRobot export API (internal)
- **[Modular Installation](./modular_installation.md)** - Plan for lightweight deployment with optional dependencies

### Detailed Component Designs

- **[Camera Interface Design](../camera_interface_design.md)** - Detailed camera interface specification
- **[Robot Interface Design](../robot_interface_design.md)** - Detailed robot interface specification
- **[LeRobot Integration](../inferencekit_lerobot_integration.md)** - LeRobot PolicyPackage integration

### Legacy Documents

- **[Physical AI Inference Framework](../../physical_ai_inference_framework.md)** - Original combined document (superseded by split documents)
- **[Universal Inference Package Design](../../inference_package_design.md)** - Original detailed design (superseded by inferencekit_design.md)

---

## Document Structure

```
docs/inference/
├── README.md                           # This file
├── overview.md                         # High-level architecture
├── inferencekit_design.md              # Generic inference package (Part I)
├── phyai_design.md                     # Physical AI framework (Part II)
├── lerobot_export_suggestions.md       # Internal: LeRobot API improvement notes
└── modular_installation.md             # Internal: Lightweight deployment plan

docs/
├── camera_interface_design.md          # Detailed camera interface
├── robot_interface_design.md           # Detailed robot interface
└── inferencekit_lerobot_integration.md # LeRobot integration

Root/
├── physical_ai_inference_framework.md  # Legacy combined document
└── inference_package_design.md         # Legacy detailed design
```

---

## Reading Guide

### For New Readers

1. Start with **[Overview](./overview.md)** to understand the high-level architecture
2. Read **[inferencekit Design](./inferencekit_design.md)** to understand the generic inference package
3. Read **[phyai Design](./phyai_design.md)** to understand the robotics framework

### For Implementation

1. Read the core design documents (inferencekit and phyai)
2. Refer to detailed component designs as needed:
   - [Camera Interface Design](../camera_interface_design.md) for camera implementation
   - [Robot Interface Design](../robot_interface_design.md) for robot implementation
   - [LeRobot Integration](../inferencekit_lerobot_integration.md) for LeRobot support

### For Integration

- **geti-action users**: See [phyai Design - geti-action Integration](./phyai_design.md#geti-action-integration)
- **LeRobot users**: See [phyai Design - LeRobot Integration](./phyai_design.md#lerobot-integration)
- **Custom framework**: See [phyai Design - Framework Extension Pattern](./phyai_design.md#framework-extension-pattern)

---

## Package Naming

### Current Status (2026-01-28)

| Package               | Proposed Name  | Status                                 |
| --------------------- | -------------- | -------------------------------------- |
| Generic Inference     | `inferencekit` | Proposed replacement for `model_api`   |
| Physical AI Framework | `phyai`        | Proposed (PyPI availability pending)   |
| Camera Interface      | TBD            | Currently `phyai.cameras` (subpackage) |

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

| Version | Date       | Changes                                                                       |
| ------- | ---------- | ----------------------------------------------------------------------------- |
| 1.2     | 2026-01-28 | Added modular installation plan                                               |
| 1.1     | 2026-01-28 | Added TwoPhaseRunner, LeRobot export suggestions, fixed LeRobot compatibility |
| 1.0     | 2026-01-28 | Initial split from combined document                                          |

---

_Last Updated: 2026-01-28_
