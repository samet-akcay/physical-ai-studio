# Design Documentation

Architecture and design documents for Geti Action — an end-to-end platform for robot AI development.

---

## Start Here

**[Strategy](./strategy.md)** — The two core architectural decisions:

1. Library-first: the library owns all components, the application is just UI/orchestration
2. Layered deployment stack: physical‑ai‑framework (universal physical‑AI engine with inference core as internal modular layer, built‑in format loaders and runners) → external plugins (exotic patterns only). Vision (model_api) can share the inference core.

---

## Library Components

Components owned by the **library** (`pip install getiaction`) — the building blocks for end-to-end robot AI.

| Component        | Document                                            | Description                                                              |
| ---------------- | --------------------------------------------------- | ------------------------------------------------------------------------ |
| Robot Interface  | [Robot Interface](./library/robot-interface.md)     | Robot ABC, leader/follower wrappers, SDK integration (LeRobot, UR, ABB)  |
| Camera Interface | [Camera Interface](./library/camera-interface.md)   | Camera ABC, invisible sharing, callbacks, capability mixins (PTZ, depth) |
| Teleoperation    | [Teleoperation API](./library/teleoperation.md)     | Leader/follower semantics, session lifecycle, safety primitives          |
| Data Collection  | [Data Collection API](./library/data-collection.md) | DatasetWriter, episode management, HF Hub upload                         |

## Deployment Stack

The inference and deployment architecture: **physical‑ai‑framework** (universal physical‑AI engine with inference core as internal modular layer, built‑in format loaders and runners) → **external plugins** (only for exotic execution patterns). Vision remains a separate domain layer (**model_api**) that can share the inference core.

| Component           | Document                                             | Description                                                        |
| ------------------- | ---------------------------------------------------- | ------------------------------------------------------------------ |
| Inference Core      | [Inference Core](./deployment/inferencekit.md)       | Domain-agnostic inference layer design (internal to framework)     |
| Deployment Engine   | [Deployment Shell](./deployment/deployment-shell.md) | physical‑ai‑framework universal engine, CLI, format loaders        |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md)       | Built‑in format loader for LeRobot manifest.json (proposed format) |

## Internal Notes

| Document                                                               | Description                                    |
| ---------------------------------------------------------------------- | ---------------------------------------------- |
| [LeRobot Export Suggestions](./internal/lerobot-export-suggestions.md) | Improvement suggestions for LeRobot export API |

---

## Document Structure

```
docs/design/
├── README.md                              # This file
├── strategy.md                            # Architecture vision & key decisions
│
├── library/                               # Library-owned components
│   ├── robot-interface.md                 # Robot ABC & SDK wrappers
│   ├── camera-interface.md                # Camera ABC & sharing
│   ├── teleoperation.md                   # Teleoperation API
│   └── data-collection.md                 # Data collection API
│
├── deployment/                            # Deployment stack
│   ├── inferencekit.md                    # Inference core layer design
│   ├── deployment-shell.md                # physical-ai-framework (CLI)
│   └── lerobot.md                         # LeRobot format loader integration
│
└── internal/
    └── lerobot-export-suggestions.md      # Internal: LeRobot API notes
```

---

## Reading Guide

### For New Readers

1. Start with **[Strategy](./strategy.md)** — understand the two core decisions
2. Read **[Inference Core](./deployment/inferencekit.md)** for the inference foundation
3. Explore library component docs for specific areas

### For Library Development

1. **[Strategy § Part 1](./strategy.md#part-1-library-first-architecture)** — ownership boundaries
2. Component design for the area you're working on:
   - [Robot Interface](./library/robot-interface.md)
   - [Camera Interface](./library/camera-interface.md)
   - [Teleoperation](./library/teleoperation.md)
   - [Data Collection](./library/data-collection.md)

### For Deployment

1. **[Strategy § Part 2](./strategy.md#part-2-deployment-stack)** — layered architecture
2. **[Inference Core](./deployment/inferencekit.md)** — domain-agnostic inference layer design
3. **[Deployment Shell](./deployment/deployment-shell.md)** — CLI and configuration

---

## Contributing

When updating these documents:

1. **Keep documents focused** — each document covers a single component
2. **Maintain cross-references** — link between related documents using relative paths
3. **Update date** — at the bottom of each document
4. **Update this README** — if you add new documents or change structure
5. **Library vs deployment** — put library-owned components in `library/`, deployment stack docs in `deployment/`

---

## Version History

| Version | Date       | Changes                                                                       |
| ------- | ---------- | ----------------------------------------------------------------------------- |
| 6.0     | 2026-02-11 | Updated to two-tier model: built‑in format loaders/runners + external plugins |
| 5.1     | 2026-02-09 | Updated deployment stack to physical‑ai‑framework as universal engine         |
| 4.0     | 2026-02-06 | Reorganized into library/ + deployment/ to reflect ownership boundaries       |
| 3.0     | 2026-02-06 | Reorganized into strategy + components/integrations/internal                  |
| 2.0     | 2026-02-05 | Restructured with physical‑ai‑framework codename                              |
| 1.0     | 2026-01-28 | Initial split from combined document                                          |

---

_Last Updated: 2026-02-11_
