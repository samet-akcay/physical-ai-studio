# Design Documentation

Architecture and design documents for Geti Action — an end-to-end platform for robot AI development.

---

## Start Here

**[Strategy](./strategy.md)** — The two core architectural decisions:

1. Library-first: the library owns all components, the application is just UI/orchestration
2. Layered deployment stack: inferencekit (base) → physical‑ai‑framework (universal physical‑AI engine with built‑in format loaders and runners) → external plugins (exotic patterns only). Vision remains model_api → inferencekit.

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

The inference and deployment architecture: **inferencekit** (base engine) → **physical‑ai‑framework** (universal physical‑AI engine with built‑in format loaders and runners) → **external plugins** (only for exotic execution patterns). Vision remains a separate domain layer (**model_api**) on inferencekit.

| Component           | Document                                             | Description                                                             |
| ------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| Inference Engine    | [inferencekit](./deployment/inferencekit.md)         | Base execution engine: RuntimeAdapter, metadata IO, base InferenceModel |
| Deployment Engine   | [Deployment Shell](./deployment/deployment-shell.md) | physical‑ai‑framework universal engine, CLI, format loaders             |
| LeRobot Integration | [LeRobot Integration](./deployment/lerobot.md)       | Built‑in format loader for LeRobot manifest.json                        |

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
│   ├── inferencekit.md                    # Base inference framework
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
2. Read **[inferencekit](./deployment/inferencekit.md)** for the inference foundation
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
2. **[inferencekit](./deployment/inferencekit.md)** — base inference framework and plugin system
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
