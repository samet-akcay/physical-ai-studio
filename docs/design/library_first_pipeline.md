# Library-First Pipeline

## Executive Summary

We need a **library-first** pipeline so teams can deploy and operate at the edge without relying on the application UI. Moving the core runtime into the library enables:

- **Edge-only deployments** (library + CLI is sufficient)
- **Consistent workflows** across CLI, scripts, and UI
- **Faster adoption** via `pip install`

The library becomes the system of record for robot, teleoperation, and data collection APIs. The application becomes glue/UI only.

---

## Architecture Overview

```
                 ┌──────────────────────────────────────────┐
                 │                APPLICATION               │
                 │  UI + orchestration (glue only)          │
                 └──────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            LIBRARY CORE                                   │
│  Robot API  │  Teleoperation API  │  Data Collection API  │  CLI Entry    │
└───────────────────────────────────────────────────────────────────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
   Robot SDKs         Teleoperation loop  LeRobot dataset
 (optional extras)    (headless)          + HF Hub upload
```

---

## High-Level API

### Python

```python
from getiaction.robots import SO101
from getiaction.teleop import TeleopSession
from getiaction.data import DatasetWriter

robot = SO101.from_config("robot.yaml")

with TeleopSession(robot=robot) as teleop:
    with DatasetWriter("./dataset", robot_type=robot.robot_type) as writer:
        for step in teleop:
            writer.add_frame(step.observation, action=step.action)
        writer.save_episode(task="stack blocks")
        writer.finalize()
        writer.push_to_hub("org/my-dataset")
```

### CLI

```bash
# Teleoperation (headless)
getiaction teleop --config teleop.yaml

# Record a dataset
getiaction record --config record.yaml

# Upload to HF Hub
getiaction upload --repo-id org/my-dataset --path ./dataset
```

---

## Key Decisions

- **Async core + sync wrapper** for Robot API (app needs async, CLI needs sync)
- **CLI uses LightningCLI/jsonargparse** (align with existing getiaction patterns)
- **Explicit connection strings only** in v1 (no discovery layer)
- **LeRobot format** required for VLA use cases

## Guardrails

- No heuristic robot detection (e.g., based on action length)
- LeRobot format only in v0.1.0; future extension via converters
- No hardware SDKs in core library (optional extras only)
- No offline buffering/async upload queue in v1

---

## Future Considerations

- Multi-arm + shared camera semantics (composition vs single robot abstraction)
- Camera device identification (`device_key` standardization)
- Long-term discovery strategy

---

## Related Documentation

- [Overview](./overview.md) - Big-picture architecture and the three-layer model
- [Data Collection API](./data_collection_api.md) - Dataset writer, metadata, HF Hub upload
- [Teleoperation API](./teleoperation_api.md) - Leader/follower semantics, lifecycle, safety
- [Deployment Shell](./deployment_shell.md) - CLI and deployment patterns for physical-ai-framework

---

_Last Updated: 2026-02-05_
