# Deployment Shell (physical‑ai‑framework)

CLI and configuration patterns for the physical‑ai‑framework deployment shell.

---

## Overview

**physical‑ai‑framework** is the codename for the deployment/inference repo. It provides a thin CLI and packaging layer while reusing getiaction's core implementation.

**Key Features:**

- CLI entrypoints for inference and deployment
- Configuration-driven workflows
- Minimal dependencies
- Production-ready packaging

---

## Architecture

```
                inferencekit
                      ▲
                      │
                  getiaction
                      ▲
                      │
           physical‑ai‑framework (shell)
                      │
                      ▼
                   CLI / Config
```

**Key principle:** physical‑ai‑framework is a thin wrapper. All core logic lives in getiaction.

---

## CLI Reference

### Run Inference

```bash
# Run policy inference on robot
phyai run \
    --model ./exports/act_policy \
    --robot robot.yaml \
    --episodes 10

# With explicit backend
phyai run \
    --model ./exports/act_policy \
    --robot robot.yaml \
    --backend openvino \
    --device CPU
```

### Serve Model

```bash
# Start inference server
phyai serve \
    --model ./exports/act_policy \
    --host 0.0.0.0 \
    --port 8080
```

### Export Policy

```bash
# Export trained policy to deployment format
phyai export \
    --checkpoint ./checkpoints/act_policy \
    --output ./exports \
    --backend onnx
```

---

## Configuration

### Deployment Configuration

```yaml
# deploy.yaml
model:
  path: ./exports/act_policy
  backend: openvino
  device: CPU

robot:
  type: so101
  port: /dev/ttyUSB0
  cameras:
    top:
      type: webcam
      index: 0

inference:
  rate_hz: 30.0
  num_episodes: 10

safety:
  action_min: -1.0
  action_max: 1.0
  velocity_limit: 0.5
```

### Running from Config

```bash
phyai run --config deploy.yaml
```

---

## Python API

For programmatic access, use getiaction directly:

```python
from getiaction.inference import InferenceModel
from getiaction.robots import SO101

# Load model and robot
policy = InferenceModel.load("./exports/act_policy", backend="openvino")
robot = SO101.from_config("robot.yaml")

# Run inference loop
with robot:
    policy.reset()
    for _ in range(1000):
        obs = robot.get_observation()
        action = policy.predict(obs)["action"]
        robot.send_action(action)
```

---

## Deployment Patterns

### Pattern 1: Edge Deployment (Jetson)

```yaml
# jetson_deploy.yaml
model:
  path: ./exports/act_policy
  backend: onnx
  device: cuda # TensorRT acceleration

robot:
  type: so101
  port: /dev/ttyUSB0

inference:
  rate_hz: 30.0
  warmup_steps: 10 # Warm up GPU
```

```bash
phyai run --config jetson_deploy.yaml
```

### Pattern 2: Intel CPU Deployment

```yaml
# intel_deploy.yaml
model:
  path: ./exports/act_policy
  backend: openvino
  device: CPU # or GPU for integrated graphics

robot:
  type: so101
  port: /dev/ttyUSB0

inference:
  rate_hz: 30.0
```

### Pattern 3: Server Deployment

```yaml
# server_deploy.yaml
model:
  path: ./exports/act_policy
  backend: onnx
  device: cuda:0

server:
  host: 0.0.0.0
  port: 8080
  workers: 1
```

```bash
phyai serve --config server_deploy.yaml
```

---

## Backend Selection Guide

| Hardware      | Recommended Backend  | Device |
| ------------- | -------------------- | ------ |
| Intel CPU     | `openvino`           | `CPU`  |
| Intel GPU     | `openvino`           | `GPU`  |
| NVIDIA GPU    | `onnx` or `tensorrt` | `cuda` |
| NVIDIA Jetson | `onnx` (TensorRT EP) | `cuda` |
| Apple Silicon | `onnx` (CoreML EP)   | `cpu`  |

---

## Installation

```bash
# Core package (inference only)
pip install physical-ai-framework

# With specific backend
pip install physical-ai-framework[openvino]
pip install physical-ai-framework[onnx-gpu]
pip install physical-ai-framework[tensorrt]

# With robot support
pip install physical-ai-framework[lerobot]
pip install physical-ai-framework[all]
```

---

## What physical‑ai‑framework Contains

**Contains:**

- CLI entrypoints (`phyai run`, `phyai serve`, `phyai export`)
- Configuration loading and validation
- Packaging and branding
- Deployment-focused documentation

**Does NOT contain:**

- Core inference implementation (lives in inferencekit)
- Robot/camera implementations (lives in getiaction)
- Training code (lives in getiaction)

---

## Related Documentation

- **[Overview](./overview.md)** - Big-picture architecture
- **[inferencekit Design](./inferencekit_design.md)** - Core inference package
- **[Robot Interface Design](./robot_interface_design.md)** - Robot interface specification

---

_Last Updated: 2026-02-05_
