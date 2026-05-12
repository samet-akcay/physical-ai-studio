# Observation Design

This document defines the canonical `Observation` type for PhysicalAI, where it lives, and how subsystems (training, runtime, benchmark) extend it without bloating the core or pulling optional dependencies into the runtime package.

## 1. Decision

`Observation` is a single canonical type, owned by the runtime distribution.

```python
from physicalai import Observation
```

```text
physicalai distribution
  physicalai/observation.py
    Observation
    ObservationConverters
    ObservationAccessors
  physicalai/converters/
    dict.py           # obs.to.dict()

physicalai-train distribution
  physicalai/train/observation.py
    to_tensor(...)
    TrainAccessor
  physicalai/data/lerobot/
    to_lerobot(...)   # obs.to.lerobot()
```

`physicalai-train` must not ship its own `physicalai/observation.py`.

## 2. Why `physicalai.observation` (Not `physicalai.data.observation`)

`Observation` flows through both pipelines:

```text
robot   -> observation -> preprocess -> model -> action
dataset -> batch       -> policy     -> loss
```

It is broader than datasets, so it does not belong under `physicalai.data`.

```text
physicalai.observation     core runtime contract
physicalai.data            datasets, datamodules, collators
physicalai.train           torch/training extensions
physicalai.runtime (or inference)        runtime execution (PolicyRuntime, evaluators)
```

`physicalai.data` may use `Observation`; it does not own it.

## 3. Core Contract

The runtime `Observation` is NumPy/Python-only and immutable.

```python
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass(frozen=True)
class Observation:
    state: np.ndarray | dict[str, np.ndarray] | None = None
    images: np.ndarray | dict[str, np.ndarray] | None = None
    action: np.ndarray | dict[str, np.ndarray] | None = None
    task: str | None = None
    timestamp: float | None = None

    info: dict[str, Any] | None = None
    extra: dict[str, Any] | None = None
```

Guarantees:

- no torch import
- stable field names
- usable by robot, runtime, inference, benchmark, and train

These are the **universal fields**. Every subsystem reads them the same way.

## 4. Core Method: `map`

`map` applies a function to each observation value while preserving structure. The name follows the pandas/Python convention: transform values, keep structure.

```python
def map(self, fn):
    def visit(value):
        if isinstance(value, dict):
            return {k: visit(v) for k, v in value.items()}
        return fn(value)

    return Observation(
        state=visit(self.state),
        images=visit(self.images),
        action=visit(self.action),
        task=self.task,
        timestamp=self.timestamp,
        info=self.info,
        extra=self.extra,
    )
```

```python
obs2 = obs.map(lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) else x)
```

`task`, `timestamp`, `info`, and `extra` are metadata and are not traversed.

## 5. Construction and Immutability

`Observation` is a frozen dataclass. It is never mutated in place. New observations are produced by constructing a new instance or by `dataclasses.replace`.

The runtime builds an observation incrementally from raw inputs:

```python
from dataclasses import replace

raw = robot.get_observation()                # dict[str, np.ndarray]
obs = Observation(state=raw["state"], images=raw["images"])

# add external cameras
obs = replace(obs, images={**(obs.images or {}), **camera_frames})

# attach runtime metadata into the extra namespace
obs = replace(
    obs,
    timestamp=clock.now(),
    extra={"frame_index": i, "episode_index": ep},
)
```

Why never mutate:

- callbacks may hold a snapshot for recording or telemetry
- async/threaded execution may reference the previous observation
- `obs.map(fn)` already returns a new observation; mutation would be inconsistent
- tests stay deterministic

A lightweight `ObservationBuilder` is acceptable when construction gets verbose, but it must produce a fresh `Observation` at `.build()`:

```python
obs = (
    ObservationBuilder()
    .with_state(raw["state"])
    .with_images(raw["images"])
    .with_camera_frames(camera_frames)
    .with_runtime_metadata(timestamp=clock.now(), frame_index=i)
    .build()
)
```

Inverse construction from external formats (e.g. LeRobot dicts) belongs in classmethod constructors or format-specific modules, not in the converter registry:

```python
Observation.from_lerobot(lerobot_dict)
Observation.from_dict(d)
```

## 6. Extension Model

Two mechanisms, both lazy and entry-point driven. Neither requires subclassing or class mutation.

| Mechanism  | Shape                   | Use for                                  |
|------------|-------------------------|------------------------------------------|
| Converter  | `obs.to.<name>(...)`    | producing a transformed observation      |
| Accessor   | `obs.<ns>.<attr/method>`| domain-specific fields and behavior      |

### 6.1 Converters

Converters take an observation and produce a transformed result. They share a single `to` namespace.

Two categories:

| Category          | Examples                          | Return type                       |
|-------------------|-----------------------------------|-----------------------------------|
| Value converters  | `numpy`, `tensor`, `jax`          | `Observation` with new array backing |
| Format converters | `lerobot`, `rlds`, `dict`         | the target format's native type   |

Both register through the same entry-point group (`physicalai.observation_converters`) and are accessed the same way:

```python
obs.to.numpy()                # Observation
obs.to.tensor(device="cuda")  # Observation (tensor-backed)
obs.to.lerobot()              # dict[str, Any]
obs.to.dict()                 # dict[str, Any]
```

The registry does not enforce a return type. Each converter's docstring declares what it returns, the same way pandas users learn that `df.to_dict()` returns a dict and `df.to_numpy()` returns an array.

```python
class ObservationConverters:
    _registry: dict = {}
    _loaded = False

    def __init__(self, observation):
        self._observation = observation

    @classmethod
    def register(cls, name, fn):
        cls._registry[name] = fn

    def __getattr__(self, name):
        self._load()
        try:
            fn = self._registry[name]
        except KeyError as e:
            raise AttributeError(f"No observation converter {name!r} is registered.") from e
        return lambda **kw: fn(self._observation, **kw)

    @classmethod
    def _load(cls):
        if cls._loaded:
            return
        from importlib.metadata import entry_points
        for ep in entry_points(group="physicalai.observation_converters"):
            cls.register(ep.name, ep.load())
        cls._loaded = True
```

On `Observation`:

```python
@property
def to(self) -> ObservationConverters:
    return ObservationConverters(self)
```

Runtime registers dependency-free converters:

```python
ObservationConverters.register("numpy", lambda obs: obs)
ObservationConverters.register("dict", lambda obs: {...})
```

Format converters that depend on optional packages (LeRobot, RLDS) live in the package that owns the format and register via entry points in that package's `pyproject.toml`.

Inverse direction (e.g. `dict -> Observation`) is **not** a converter. It belongs in classmethod constructors (see §5).

### 6.2 Accessors

Accessors expose domain-specific fields and behavior under a namespace. This mirrors pandas (`df["x"].dt`, `.str`, `.cat`) and xarray accessors.

```python
class ObservationAccessors:
    _registry: dict = {}
    _loaded = False

    @classmethod
    def register(cls, name, accessor_cls):
        cls._registry[name] = accessor_cls

    @classmethod
    def get(cls, name, observation):
        cls._load()
        try:
            accessor_cls = cls._registry[name]
        except KeyError as e:
            raise AttributeError(
                f"No observation accessor {name!r} is registered. "
                f"Is the relevant package installed?"
            ) from e
        return accessor_cls(observation)

    @classmethod
    def _load(cls):
        if cls._loaded:
            return
        from importlib.metadata import entry_points
        for ep in entry_points(group="physicalai.observation_accessors"):
            cls.register(ep.name, ep.load())
        cls._loaded = True
```

On `Observation`:

```python
def __getattr__(self, name):
    return ObservationAccessors.get(name, self)
```

## 7. Where Things Belong

The base type stays minimal. Everything else lives in a namespace.

Put a field/method in **core** when:

- every subsystem reads it
- it is part of the fundamental observation contract
- it has no optional dependency

Put it in a **namespace accessor** when:

- only one subsystem produces or consumes it
- it depends on optional packages (torch, gym, ...)
- it is metadata about *how* the observation was produced

```python
# core
obs.state
obs.images
obs.action
obs.task
obs.timestamp
obs.map(fn)
obs.to.numpy()

# train-only (installed by physicalai-train)
obs.train.episode_index
obs.train.frame_index
obs.train.dataset_index
obs.to.tensor(device="cuda")        # converter, shared shortcut

# runtime-only (only if runtime needs its own metadata)
obs.runtime.latency_ms
obs.runtime.source_robot
obs.runtime.validate(schema)

# benchmark-only
obs.benchmark.task_id
obs.benchmark.success
```

`obs.runtime` is initially not required. Runtime code uses core fields directly. The namespace exists only when runtime needs its own attached metadata or behavior.

## 8. Train Extension

`physicalai-train` ships an accessor and a converter, both registered via entry points.

```python
# physicalai-train distribution
# physicalai/train/observation.py

import numpy as np
import torch
from physicalai import Observation

def to_tensor(obs: Observation, device="cpu") -> Observation:
    return obs.map(
        lambda x: torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x
    )

class TrainAccessor:
    def __init__(self, observation: Observation):
        self._obs = observation

    @property
    def episode_index(self) -> int | None:
        return (self._obs.extra or {}).get("episode_index")

    @property
    def frame_index(self) -> int | None:
        return (self._obs.extra or {}).get("frame_index")

    def to_tensor(self, device="cpu") -> Observation:
        return to_tensor(self._obs, device=device)
```

```toml
[project.entry-points."physicalai.observation_converters"]
tensor = "physicalai.train.observation:to_tensor"

[project.entry-points."physicalai.observation_accessors"]
train = "physicalai.train.observation:TrainAccessor"
```

User code:

```python
from physicalai import Observation

obs = Observation(state=np.array([1.0, 2.0], dtype=np.float32))

# always available
obs.to.numpy()

# available when physicalai-train is installed
obs.to.tensor(device="cuda")
obs.train.episode_index
obs.train.to_tensor(device="cuda")
```

If `physicalai-train` is not installed, `obs.train` and `obs.to.tensor` raise `AttributeError` with a clear message.

## 9. Usage by Subsystem

```python
# Runtime: core only
def step(obs: Observation):
    action = policy(obs.state, obs.images)
    return action

# Training: core + train
def training_step(obs: Observation):
    x = obs.to.tensor()
    idx = obs.train.episode_index
    return loss_fn(x, idx)

# Benchmark: core + benchmark
def evaluate(obs: Observation):
    if obs.benchmark.success:
        ...
```

Each subsystem touches only what it needs. The base type does not change.

## 10. Composition for Richer Payloads

When training needs more than one observation (e.g. a target frame for supervision), use composition rather than extending the accessor.

```python
@dataclass(frozen=True)
class TrainingSample:
    observation: Observation
    target: Observation | None = None
    weight: float = 1.0
```

Use accessors for **metadata about an observation**. Use composition for **structures that contain multiple observations or non-observation payloads**.

## 11. Why Not Other Approaches

| Approach                         | Problem                                                  |
|----------------------------------|----------------------------------------------------------|
| `class TorchObservation(Observation)` | two types in the wild, return-type drift           |
| `to_tensor` on the base class    | runtime imports torch, base class grows over time        |
| Free function `to_tensor(obs)`   | poor discoverability, users guess module locations       |
| Monkey-patch `Observation` from train | hidden mutation, type checkers blind, collisions    |
| `__getattr__` flattening accessors onto the base | silent collisions, origin of methods unclear |

The chosen design keeps one canonical type, isolates optional dependencies, and makes the source of every method explicit.

## 12. Why Namespaces Are a Feature

The two-character cost of `obs.train.foo` over `obs.foo` buys:

- explicit ownership (you know which package adds it)
- collision-free extension across packages
- clear failure mode when a package is missing
- type checker and IDE visibility on the accessor class
- safe addition of new accessors without breaking user code

This is the same trade pandas chose with `.dt`, `.str`, `.cat`.

## 13. Namespace Package Rule

Allowed:

```text
physicalai distribution        physicalai/observation.py
physicalai-train distribution  physicalai/train/observation.py
```

Not allowed:

```text
physicalai distribution        physicalai/observation.py
physicalai-train distribution  physicalai/observation.py   # collision
```

The top-level `physicalai/__init__.py` uses `pkgutil.extend_path` so subpackages from multiple distributions resolve under one namespace.

## 14. Package Sketch

```text
physicalai distribution
  physicalai/__init__.py
    __path__ = pkgutil.extend_path(__path__, __name__)
    from physicalai.observation import Observation

  physicalai/observation.py
    Observation
    ObservationConverters
    ObservationAccessors

  physicalai/converters/
    dict.py                       # obs.to.dict()

physicalai-train distribution
  physicalai/train/observation.py
    to_tensor                     # obs.to.tensor()
    TrainAccessor                 # obs.train.*

  physicalai/data/lerobot/
    to_lerobot                    # obs.to.lerobot()
    from_lerobot                  # Observation.from_lerobot(...)

  physicalai/data/
    datasets, datamodules, collators

  pyproject.toml
    [project.entry-points."physicalai.observation_converters"]
    tensor  = "physicalai.train.observation:to_tensor"
    lerobot = "physicalai.data.lerobot:to_lerobot"

    [project.entry-points."physicalai.observation_accessors"]
    train = "physicalai.train.observation:TrainAccessor"
```

## 15. Non-Goals

- torch dependency in `physicalai`
- `TorchObservation` or any subclass of `Observation`
- duplicate `physicalai/observation.py` in train
- promoting accessor methods onto the base class
- in-place mutation of `Observation`
- making `physicalai.data` own the runtime observation contract

## 16. Summary

```text
Canonical type:      physicalai.Observation
Implementation:      physicalai.observation
Runtime deps:        numpy / python only
Extension:           converters (obs.to.X) + accessors (obs.<ns>.X)
Discovery:           importlib.metadata entry points (lazy)
Train install:       adds obs.to.tensor() and obs.train.*
```
