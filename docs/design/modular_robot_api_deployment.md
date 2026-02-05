# Modular Robot API & Deployment

## Executive Summary

We need a **modular Robot API** that serves both **data collection** and **real-time deployment**. By moving to a library-first, modular Robot API with optional extras, we can:

- Support **edge deployment** via library + `InferenceModel` only
- Keep installation lightweight while enabling robot-specific SDKs
- Provide a consistent contract for teleop, data collection, and inference

The library becomes the home of the Robot API. Robot implementations are modularized by brand or SDK.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                      LIBRARY CORE                    │
│  Robot API  │  Internal Mapping  │  InferenceModel    │
└──────────────────────────────────────────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
   Robot Extras       Format targets     Edge deployment
(LeRobot, UR, etc.)   (e.g., LeRobot)     (library-only)
```

### Modular Install

```bash
pip install getiaction                # core API only
pip install getiaction[lerobot]       # LeRobot robots
pip install getiaction[trossen]       # Trossen robots
pip install getiaction[ur]            # UR robots (future)
```

---

## High-Level API

### Inference Deployment

```python
from getiaction.inference import InferenceModel
from getiaction.robots import TrossenWidowXAI

robot = TrossenWidowXAI.from_config("robot.yaml")
policy = InferenceModel.load("./exports/policy")

with robot:
    for step in range(max_steps):
        obs = robot.get_observation()
        action = policy.select_action(obs)
        robot.send_action(action)
        if should_stop():
            break
```

### Data Collection

```python
from getiaction.data import DatasetWriter
from getiaction.robots import SO101

robot = SO101.from_config("robot.yaml")

with DatasetWriter(path="./dataset", robot=robot, format="lerobot") as writer:
    # ... collect frames and actions ...
    writer.save_episode(task="demo")
    writer.finalize()
```

---

## Robot API

### Core Interface

```python
class Robot:
    robot_type: RobotType
    observation_features: dict
    action_features: dict

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> Observation: ...
    def send_action(self, action) -> None: ...
```

Tools operate on the stable interface, while robot-specific capability differences are expressed through feature descriptors and `robot_type`.

---

## Key Decisions

- **Robot API in library** with optional extras for SDKs
- **Per-robot schemas with internal mappers** (preserve fidelity)
- **User-facing API is format-centric** (e.g., `format="lerobot"`)
- **Single contract for inference + collection** (no separate runtime APIs)

### v0.1.0 Scope

- Robots: **SO101 + Trossen** only
- Other brands post-v0.1.0 via modular extras

## Guardrails

- No SDK dependencies in core library
- No user-facing adapter configuration required
- No application dependency for edge deployment

---

## Future Considerations

- Timing guarantees for control loop rate
- Unified profile schema for common workflows

---

## Related Documentation

- [Overview](./overview.md) - Big-picture architecture and the three-layer model
- [Robot Interface Design](./robot_interface_design.md) - Detailed robot interface specification
- [Deployment Shell](./deployment_shell.md) - CLI and deployment patterns for physical-ai-framework
- [Library-First Pipeline](./library_first_pipeline.md) - Library-first design for teleop and data collection

---

_Last Updated: 2026-02-05_
