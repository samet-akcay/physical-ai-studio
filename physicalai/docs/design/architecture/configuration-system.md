# Configuration System Design

## Goal

PhysicalAI should support the same workflows from Python, YAML, CLI, and Studio
APIs.

The configuration system should make these paths equivalent:

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

```bash
physicalai-train fit --config training.yaml
```

```python
Trainer.from_config(studio_payload_dict).fit()
```

```python
config = trainer.to_config()
config.to_yaml("training.reproduced.yaml")
```

The same model should also work for inference and robot runtime:

```python
model = InferenceModel.from_config("inference.yaml")
runtime = PolicyRuntime.from_config("runtime.yaml")
```

```bash
physicalai infer --config inference.yaml
physicalai run --config runtime.yaml
```

## Direction

The configuration backbone is **jsonargparse + dataclasses / Pydantic**, not a
custom config layer.

- Each class is its own schema. Constructor signatures (with dataclass /
  Pydantic args where appropriate) are the source of truth.
- jsonargparse handles YAML loading, CLI parsing, `class_path` / `init_args`
  dispatch, and recursive instantiation.
- `physicalai.config` provides a thin convenience surface on top:
  - `FromConfig` / `@from_config` for class-level `cls.from_config(...)` sugar.
  - `instantiate_obj` / `instantiate_obj_from_dict` for programmatic use.
- Workflow orchestrators (`Trainer`, `PolicyRuntime`, `InferenceModel`) own
  `from_config(...)` and `to_config()` entry points. Internally they wrap a
  jsonargparse parser.
- `library/src/physicalai/config/base.py::Config` shrinks to a checkpoint
  serialization helper (`to_dict` / `from_dict` for torch `weights_only=True`
  round-tripping). `to_jsonargparse` / `save` / `load` are removed.
- `ComponentSpec` stays scoped to `physicalai.inference.manifest` and is used
  only inside `Manifest`. It is not promoted as a public training / runtime
  primitive; jsonargparse's native `class_path` / `init_args` covers that
  ground.

## Big Picture

The system has three data layers and one execution layer.

```text
Class schema (constructor + dataclass/Pydantic args)
  typed constructor arguments for one class

Workflow config (YAML / dict / jsonargparse Namespace)
  user-authored workflow graph before execution

Manifest (exported package metadata)
  concrete artifacts + runtime descriptors after build/export

Orchestrator
  live object that consumes configs/manifests and runs the workflow
```

Examples:

```text
Pi05 / Pi05Config        -> constructor args for Pi05
TrainingConfig           -> model + data + trainer (YAML or dataclass)
Manifest                 -> exported artifacts + runtime metadata
Trainer                  -> orchestrates training
InferenceModel           -> loads model artifacts and computes actions
PolicyRuntime            -> runs robot control loop
```

Passive data objects do not execute workflows. Orchestrators do.

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

## Core Concepts

### Class Schemas

A class is its own schema. There is no requirement to wrap every class in a
`Config` object.

```python
class Pi05:
    def __init__(self, chunk_size: int = 50, n_action_steps: int = 50) -> None:
        if n_action_steps > chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
```

YAML form (jsonargparse subclass syntax):

```yaml
class_path: physicalai.policies.pi05.Pi05
init_args:
  chunk_size: 50
  n_action_steps: 50
```

When a constructor takes a typed structured argument, a dataclass (or Pydantic
model) is used purely as the type for that argument.

```python
@dataclass
class Pi05Hyperparams:
    chunk_size: int = 50
    n_action_steps: int = 50

class Pi05:
    def __init__(self, hparams: Pi05Hyperparams) -> None:
        ...
```

jsonargparse handles parsing nested dicts / YAML into the dataclass. Validation
lives in `__init__` / `__post_init__`.

There is no project-wide `Config` base class for typed constructor schemas.

### FromConfig Sugar

`FromConfig` (and `@from_config`) provide ergonomic class-level constructors
for code that has a known target class.

```python
class Pi05(FromConfig):
    def __init__(self, chunk_size: int = 50, n_action_steps: int = 50) -> None:
        ...
```

Usage:

```python
Pi05.from_config({"chunk_size": 50})
Pi05.from_config("pi05.yaml")
Pi05.from_dict({"class_path": "pkg.Pi05Plus", "init_args": {...}})
Pi05.from_pydantic(cfg)
Pi05.from_dataclass(cfg)
```

These remain in `library/src/physicalai/config/mixin.py` and are used for
LeRobot wrappers, `Pi05`, `SmolVLA`, and similar classes. They are convenience,
not infrastructure: anything they do can also be done by constructing a
jsonargparse parser explicitly.

Their internals may later be reimplemented on top of jsonargparse so the
project has a single instantiation backend with two surfaces:

```text
jsonargparse parser     -> CLI + Trainer.from_config(...)
FromConfig classmethods -> cls.from_config(...) sugar
```

Until then, `instantiate_obj_from_dict` (in `physicalai.config.instantiate`)
is the shared recursion / dispatch helper.

### Checkpoint Config Helper

The remaining role of `library/src/physicalai/config/base.py::Config` is
checkpoint serialization for torch `load(..., weights_only=True)`.

```python
class Config:
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self: ...
```

`to_jsonargparse`, `save`, and `load` are removed. YAML I/O goes through
jsonargparse (`parser.dump`, `parser.parse_path`). If no class needs the
checkpoint round trip beyond `dataclasses.asdict` / `cls(**data)`, this base
can be dropped entirely.

### Workflow Configs

A workflow config is a user-authored configuration before execution. It is
expressed in YAML (or an equivalent dict) and parsed by jsonargparse.

Training config:

```yaml
# training.yaml
trainer:
  class_path: physicalai.train.Trainer
  init_args:
    max_epochs: 10
    accelerator: gpu
    devices: 1
    callbacks:
      - class_path: physicalai.train.callbacks.ProgressCallback
        init_args: {}

model:
  class_path: physicalai.policies.pi05.Pi05
  init_args:
    chunk_size: 50
    n_action_steps: 50

data:
  class_path: physicalai.data.LeRobotDataModule
  init_args:
    repo_id: physical-ai/example
```

Python equivalent (preferred entry point):

```python
Trainer.from_config("training.yaml").fit()
```

Programmatic equivalent (dict form, e.g. from Studio):

```python
Trainer.from_config(
    {
        "trainer": {
            "class_path": "physicalai.train.Trainer",
            "init_args": {"max_epochs": 10, "callbacks": [...]},
        },
        "model": {
            "class_path": "physicalai.policies.pi05.Pi05",
            "init_args": {"chunk_size": 50},
        },
        "data": {
            "class_path": "physicalai.data.LeRobotDataModule",
            "init_args": {"repo_id": "physical-ai/example"},
        },
    }
).fit()
```

A typed `TrainingConfig` dataclass may be added later as a Python convenience
for SDK users, but it is not required by the architecture. The canonical
representation is the YAML / dict shape that jsonargparse parses.

Runtime config:

```yaml
# runtime.yaml
runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args:
        port: /dev/ttyACM0
    model:
      class_path: physicalai.inference.InferenceModel
      init_args:
        path: ./exports/act_policy
    execution:
      class_path: physicalai.runtime.SyncExecution
      init_args:
        mode: chunk
```

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

Each orchestrator owns its workflow config shape. There is no shared
`WorkflowConfig` base.

### Manifest

A manifest describes an exported package and lives in
`physicalai.inference.manifest`. It is the only place `ComponentSpec` is used
as a public type.

```yaml
format: policy_package
version: "1.0"

policy:
  name: pi05
  source:
    class_path: physicalai.policies.pi05.Pi05

model:
  artifacts:
    openvino: pi05.xml
  runner:
    type: action_chunking
    chunk_size: 50
  preprocessors:
    - type: normalize
      artifact: stats.safetensors

hardware:
  robots:
    - name: main
      type: SO101
```

Workflow config and manifest are related but not identical.

```text
Workflow config
  desired workflow before running/building/exporting

Manifest
  concrete exported package after build/export
```

The manifest can also act as an inference-time config because it already
contains the necessary `class_path` / registry data:

```python
model = InferenceModel.from_config("export/manifest.json")
```

Schema convergence with workflow configs may happen later. Initially the
boundary is kept clear: configs express intent, manifests describe artifacts.

## End-to-End Pipelines

### Training From YAML

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

Internally:

```python
parser = _build_fit_parser()       # jsonargparse parser
cfg = parser.parse_path("training.yaml")
instantiated = parser.instantiate_classes(cfg)

trainer = instantiated.trainer
model = instantiated.model
data = instantiated.datamodule

trainer._configured_model = model
trainer._configured_datamodule = data
return trainer
```

`fit()` then defaults to the bound `model` / `datamodule` when not given
explicit arguments.

### Training From CLI

```bash
physicalai-train fit --config training.yaml
physicalai-train fit \
  --trainer physicalai.train.Trainer \
  --trainer.max_epochs 10 \
  --model physicalai.policies.pi05.Pi05 \
  --model.chunk_size 50 \
  --data physicalai.data.LeRobotDataModule \
  --data.repo_id physical-ai/example
```

The CLI uses the same parser as `Trainer.from_config`. There is one definition
of the training schema.

### Training From Studio

The application layer converts the API payload to the dict shape
`Trainer.from_config` accepts.

```python
def handle_train_job(payload: TrainJobPayload) -> None:
    config = studio_payload_to_training_config(payload)
    Trainer.from_config(config).fit()
```

The worker should not manually assemble model, data module, callbacks, and
trainer if the library can own that orchestration.

```text
Studio payload
    -> application mapper (dict)
    -> Trainer.from_config(dict)
    -> Trainer.fit
```

### Training Round Trip

For ordinary components, `to_config()` describes that component. For
orchestrators, `to_config()` returns the highest-level reproducible workflow
config available.

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()

config = trainer.to_config()
config.to_yaml("training.reproduced.yaml")
```

Expected behavior:

```python
Trainer(max_epochs=10).to_config()
# returns trainer-only config (model/data omitted)

Trainer.from_config("training.yaml").to_config()
# returns full training config (trainer + model + data)
```

The serialized output uses the same jsonargparse-compatible
`class_path` / `init_args` shape as the input YAML.

### Inference From Manifest

```python
model = InferenceModel.from_config("export/manifest.json")
action = model.select_action(observation)
```

Internal shape:

```python
manifest = Manifest.load("export/manifest.json")
runner = instantiate_component(manifest.model.runner)
pre = [instantiate_component(s) for s in manifest.model.preprocessors]
post = [instantiate_component(s) for s in manifest.model.postprocessors]

model = InferenceModel(
    artifacts=manifest.model.artifacts,
    runner=runner,
    preprocessors=pre,
    postprocessors=post,
)
```

`instantiate_component` (already in `physicalai.inference.component_factory`)
handles registry mode (`type: ...`) and direct class mode (`class_path: ...`).

### Robot Runtime From YAML

```python
PolicyRuntime.from_config("runtime.yaml").run(duration_s=60)
```

```bash
physicalai run --config runtime.yaml --duration-s 60
```

## API Surface

### FromConfig

`FromConfig` is class-level construction sugar.

```python
class FromConfig:
    @classmethod
    def from_config(cls, config) -> Self: ...
    @classmethod
    def from_yaml(cls, path) -> Self: ...
    @classmethod
    def from_dict(cls, data) -> Self: ...
    @classmethod
    def from_pydantic(cls, cfg) -> Self: ...
    @classmethod
    def from_dataclass(cls, cfg) -> Self: ...
```

Used by:

- LeRobot wrappers (noisy kwargs + YAML files).
- `Pi05`, `SmolVLA`, and similar policy classes.
- Any orchestrator that wants a single ergonomic Python entry point.

Orchestrators may override `from_config` to bind subcomponents:

```python
class Trainer(FromConfig):
    @classmethod
    def from_config(cls, config) -> "Trainer":
        parser = _build_fit_parser()
        cfg = _parse(parser, config)            # path / dict / Namespace
        instantiated = parser.instantiate_classes(cfg)
        trainer = instantiated.trainer
        trainer._configured_model = instantiated.model
        trainer._configured_datamodule = instantiated.datamodule
        return trainer
```

`fit()` remains the public execution API:

```python
trainer.fit(model=model, datamodule=data)
Trainer.from_config("training.yaml").fit()
```

### ToConfig

`to_config()` is opt-in serialization.

```python
class Trainer:
    def to_config(self) -> TrainingConfig | dict[str, Any]: ...
```

Objects decide what state is reproducible constructor config and what state is
runtime-only. Outbound serialization is explicit; inbound instantiation is
generic.

For policies, export goes through the existing `policy.export(...)` path and
produces a manifest, not a workflow config.

```python
policy.export("export/", backend="openvino")
```

### Instantiation Helpers

`physicalai.config` exposes:

```python
from physicalai.config import (
    FromConfig,
    from_config,           # decorator form
    instantiate_obj,       # programmatic single object
)
```

These wrap jsonargparse / `instantiate_obj_from_dict`. Users with a known
target class can stay inside this surface; users authoring full workflows
should go through the orchestrator's `from_config`.

## CLI Parity

The CLI uses the same schema as the Python API.

```bash
physicalai run --config runtime.yaml
```

```python
PolicyRuntime.from_config("runtime.yaml").run()
```

These construct the same runtime.

`physicalai` is the torch-free runtime package. Its CLI depends on
`jsonargparse` but must not depend on torch or Lightning.

```text
physicalai
  infer
  run
  serve
  inspect-manifest

physicalai-train (library distribution)
  fit
  validate
  test
  predict
  export
```

Training commands plug into the runtime CLI via entry points, but importing
`physicalai` alone does not import training dependencies.

```toml
[project.entry-points."physicalai.cli.subcommands"]
run = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"

[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.train.cli:register_fit"
```

## Package Boundaries

```text
physicalai/                                    (torch-free)
  config/
    FromConfig
    from_config (decorator)
    instantiate_obj
    instantiate_obj_from_dict
  inference/
    manifest.py            (ComponentSpec, Manifest)
    component_factory.py   (instantiate_component)
    InferenceModel
  runtime/
    PolicyRuntime
  cli/                     (jsonargparse-based)

library/                                       (torch + Lightning)
  config/
    base.py                (Config: to_dict/from_dict for checkpoints)
    instantiate.py
    mixin.py               (FromConfig re-export / training-side helpers)
  train/
    Trainer
    cli.py                 (jsonargparse fit/validate/test/predict)
  export/
    manifest builders
```

`ComponentSpec` stays in `physicalai.inference.manifest`. It is not re-exported
from `physicalai.config`.

## Design Rules

1. Constructor signatures (with dataclass / Pydantic args) are the schema.
2. jsonargparse is the parsing + instantiation backbone for YAML, CLI, and
   dict inputs.
3. `FromConfig` and `@from_config` are Python convenience for known target
   classes.
4. Orchestrators (`Trainer`, `PolicyRuntime`, `InferenceModel`) own
   `from_config(...)` and `to_config()`.
5. Workflow YAML uses jsonargparse subclass syntax (`class_path` /
   `init_args`).
6. `ComponentSpec` is internal to `Manifest`. Do not promote it as a public
   training / runtime config primitive.
7. `Config` base (library side) is checkpoint-only or removed.
8. Keep torch and Lightning out of the `physicalai` runtime package.
9. Inbound config is generic; outbound config is explicit.
10. Application payload translation lives in the application layer; reusable
    workflow logic lives in the library / runtime layer.

## Migration Plan

### Phase 1: Shrink `Config` Base

- Remove `to_jsonargparse`, `save`, `load` from
  `library/src/physicalai/config/base.py`.
- Keep `to_dict` / `from_dict` only if needed for checkpoint round-tripping;
  otherwise drop the class entirely.
- Update `library/tests/unit/config/test_config.py` to drop tests for removed
  methods.

### Phase 2: `Trainer.from_config` / `to_config`

- Implement `_build_fit_parser()` in `library/src/physicalai/train/trainer.py`
  using jsonargparse, exposing `trainer`, `model`, `datamodule`.
- Implement `Trainer.from_config(source: str | Path | dict)` wrapping that
  parser.
- Implement `Trainer.to_config()` returning the highest-level reproducible
  config: full workflow when `from_config` constructed it, trainer-only
  otherwise.
- `fit()` defaults to bound `model` / `datamodule` when none is passed.

### Phase 3: Studio Integration

- Add `studio_payload_to_training_config(payload) -> dict` mapper in the
  application layer (or `TrainingConfig.from_studio_payload(...)` if a typed
  dataclass is introduced).
- Refactor `application/backend/src/workers/training_worker.py` to call
  `Trainer.from_config(config).fit()`.

### Phase 4: CLI

- Add jsonargparse-based CLI entry points in `physicalai/`
  (`run`, `serve`, `infer`, `inspect-manifest`).
- Add `physicalai-train` CLI in `library/`
  (`fit`, `validate`, `test`, `predict`, `export`) sharing the parsers used by
  `Trainer.from_config`.

### Phase 5: Runtime / Inference Configs

- Implement `PolicyRuntime.from_config(...)` and
  `InferenceModel.from_config(...)` using the same jsonargparse pattern.
- Confirm `Manifest` can be loaded directly by `InferenceModel.from_config`.

### Phase 6: Optional Typed Workflow Configs

- Introduce typed `TrainingConfig` / `RuntimeConfig` / `InferenceConfig`
  dataclasses only when programmatic SDK ergonomics or Studio mapping require
  them.
- These wrap the same jsonargparse parser; YAML / dict remains canonical.

## Open Questions

- Does any existing class need `Config.to_dict` / `from_dict` for checkpoint
  round-tripping, or can `Config` be removed entirely?
- Should `FromConfig` internals be reimplemented on top of jsonargparse so
  there is one instantiation backend, or stay on `instantiate_obj_from_dict`?
- Should `Trainer.to_config()` return a typed dataclass or the raw
  jsonargparse-compatible dict?
- Where does `studio_payload_to_training_config` live: application layer only,
  or as a `Trainer.from_studio_payload(...)` classmethod in the library?

The conservative default is: jsonargparse + dataclasses as the backbone,
`FromConfig` as convenience, `ComponentSpec` scoped to `Manifest`, and typed
workflow configs introduced only when a concrete need appears.
