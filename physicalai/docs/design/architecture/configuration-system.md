# Configuration System Design

## Goal

PhysicalAI should support the same workflows from Python, YAML, CLI, and Studio APIs.

The configuration system should make these paths equivalent:

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()
```

```bash
physicalai-train fit --config training.yaml
```

```python
config = TrainingConfig.from_studio_payload(payload)
Trainer.from_config(config).fit()
```

```python
config = trainer.to_config()
config.to_yaml("training.yaml")
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

## Big Picture

The system has four data layers and one execution layer.

```text
Config
  typed constructor arguments for one class

ComponentSpec
  generic target + args for one instantiable component

Workflow config
  user-authored workflow graph before execution

Manifest
  exported package description after build/export

Orchestrator
  live object that consumes specs and runs the workflow
```

Examples:

```text
Pi05Config             -> constructor args for Pi05
ComponentSpec          -> class_path + init_args for Pi05
TrainingConfig         -> model + data + trainer
Manifest               -> exported artifacts + runtime metadata
Trainer                -> orchestrates training
InferenceModel         -> loads model artifacts and computes actions
PolicyRuntime          -> runs robot control loop
```

Passive data objects should not execute workflows. Orchestrators execute workflows.

```python
config = TrainingConfig.load("training.yaml")
trainer = Trainer.from_config(config)
trainer.fit()
```

## Core Concepts

### Config

`Config` is a typed constructor-argument schema for one class.

```python
@dataclass
class Pi05Config(Config):
    chunk_size: int = 50
    n_action_steps: int = 50

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")
```

It represents only the `init_args` payload:

```yaml
chunk_size: 50
n_action_steps: 50
```

It does not decide which class to instantiate.

```python
cfg = Pi05Config(chunk_size=50)
policy = instantiate_obj(cfg, target_cls=Pi05)
```

This separation keeps typed constructor validation separate from dispatch.

### ComponentSpec

`ComponentSpec` is the generic descriptor for one instantiable component.

The runtime package already has this concept in `physicalai.inference.manifest.ComponentSpec`:

```python
class ComponentSpec(BaseModel):
    type: str = ""
    class_path: str = ""
    init_args: dict[str, Any] = Field(default_factory=dict)
```

It supports two resolution modes.

Registry mode:

```yaml
type: action_chunking
chunk_size: 50
n_action_steps: 50
```

Direct class mode:

```yaml
class_path: physicalai.inference.runners.ActionChunking
init_args:
  chunk_size: 50
  n_action_steps: 50
```

`class_path` takes precedence when both are present.

The same shape should be used across training, inference, runtime, manifests, and CLI schemas.

```python
spec = ComponentSpec(
    class_path="physicalai.policies.pi05.Pi05",
    init_args={"chunk_size": 50, "n_action_steps": 50},
)

policy = instantiate_component(spec)
```

Typed configs can feed a component spec:

```python
spec = ComponentSpec.from_config(
    target=Pi05,
    config=Pi05Config(chunk_size=50, n_action_steps=50),
)
```

Equivalent output:

```yaml
class_path: physicalai.policies.pi05.Pi05
init_args:
  chunk_size: 50
  n_action_steps: 50
```

`ComponentSpec` is data. Instantiation is a separate operation.

```text
ComponentSpec      = what should be built
instantiate_*      = build it now
```

### Workflow Config

A workflow config is a user-authored configuration before execution. It is sometimes useful to call this a recipe, but the public API does not need a separate `Recipe` suffix.

Training config:

```yaml
model:
  class_path: physicalai.policies.pi05.Pi05
  init_args:
    chunk_size: 50
    n_action_steps: 50

data:
  class_path: physicalai.data.LeRobotDataModule
  init_args:
    repo_id: physical-ai/example

trainer:
  class_path: physicalai.train.Trainer
  init_args:
    max_epochs: 10
    accelerator: gpu
    devices: 1
    callbacks:
      - class_path: physicalai.train.callbacks.ProgressCallback
        init_args: {}
```

Python equivalent:

```python
config = TrainingConfig(
    model=ComponentSpec.from_config(Pi05, Pi05Config(chunk_size=50)),
    data=ComponentSpec(
        class_path="physicalai.data.LeRobotDataModule",
        init_args={"repo_id": "physical-ai/example"},
    ),
    trainer=ComponentSpec(
        class_path="physicalai.train.Trainer",
        init_args={
            "max_epochs": 10,
            "accelerator": "gpu",
            "devices": 1,
            "callbacks": [
                ComponentSpec(
                    class_path="physicalai.train.callbacks.ProgressCallback",
                    init_args={},
                )
            ],
        },
    ),
)
```

Runtime config:

```yaml
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

Python equivalent:

```python
config = RuntimeConfig(
    runtime=ComponentSpec(
        class_path="physicalai.runtime.PolicyRuntime",
        init_args={
            "fps": 30,
            "robot": ComponentSpec(
                class_path="physicalai.robot.so101.SO101",
                init_args={"port": "/dev/ttyACM0"},
            ),
            "model": ComponentSpec(
                class_path="physicalai.inference.InferenceModel",
                init_args={"path": "./exports/act_policy"},
            ),
            "execution": ComponentSpec(
                class_path="physicalai.runtime.SyncExecution",
                init_args={"mode": "chunk"},
            ),
        },
    )
)
```

Workflow configs are domain-specific. `ComponentSpec` is generic.

```text
TrainingConfig     uses ComponentSpec for model, data, trainer
InferenceConfig    uses ComponentSpec for model, runner, processors
RuntimeConfig      uses ComponentSpec for runtime, robot, cameras, execution
```

### Manifest

A manifest describes an exported package.

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

An export pipeline can convert a trained policy plus export arguments into a manifest.

```python
policy = Trainer.from_config(training_config).fit()
policy.export("export/", backend="openvino")

manifest = Manifest.load("export/manifest.json")
```

The manifest can also act as an inference-time config because it already contains runtime component specs.

```python
model = InferenceModel.from_config("export/manifest.json")
action = model.select_action(observation)
```

Over time, workflow config and manifest schemas can converge where it improves reuse. The initial boundary should stay clear: configs express intent, manifests describe exported artifacts.

## End-to-End Pipelines

### Training From YAML

```yaml
# training.yaml
model:
  class_path: physicalai.policies.pi05.Pi05
  init_args:
    chunk_size: 50

data:
  class_path: physicalai.data.LeRobotDataModule
  init_args:
    repo_id: physical-ai/example

trainer:
  class_path: physicalai.train.Trainer
  init_args:
    max_epochs: 10
    callbacks:
      - class_path: physicalai.train.callbacks.ProgressCallback
        init_args: {}
```

```python
config = TrainingConfig.load("training.yaml")

model = instantiate_component(config.model)
data = instantiate_component(config.data)
trainer = instantiate_component(config.trainer)

trainer.fit(model=model, datamodule=data)
```

Public API:

```python
Trainer.from_config("training.yaml").fit()
```

### Training From Studio

The application layer maps API payloads into library training configs.

```python
def handle_train_job(payload: TrainJobPayload) -> None:
    config = TrainingConfig.from_studio_payload(payload)
    Trainer.from_config(config).fit()
```

The worker should not manually assemble model, data module, callbacks, and trainer if the library can own that orchestration.

```text
Studio payload
    -> application mapper
    -> TrainingConfig
    -> Trainer.from_config
    -> Trainer.fit
```

### Training Round Trip

For ordinary components, `to_config()` describes that component. For orchestrators, `to_config()` should return the highest-level reproducible workflow config available.

```python
trainer = Trainer.from_config("training.yaml")
trainer.fit()

config = trainer.to_config()
config.to_yaml("training.reproduced.yaml")
```

Expected behavior:

```python
Trainer(max_epochs=10).to_config()
# TrainingConfig(trainer=..., model=None, data=None)

Trainer.from_config("training.yaml").to_config()
# TrainingConfig(trainer=..., model=..., data=...)
```

When `model` or `data` is unavailable, the returned config can omit those sections during serialization.

### Inference From Manifest

```python
model = InferenceModel.from_config("export/manifest.json")
action = model.select_action(observation)
```

Internal shape:

```python
manifest = Manifest.load("export/manifest.json")

runner = instantiate_component(manifest.model.runner)
pre = [instantiate_component(spec) for spec in manifest.model.preprocessors]
post = [instantiate_component(spec) for spec in manifest.model.postprocessors]

model = InferenceModel(
    artifacts=manifest.model.artifacts,
    runner=runner,
    preprocessors=pre,
    postprocessors=post,
)
```

### Robot Runtime From YAML

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
runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run(duration_s=60)
```

CLI:

```bash
physicalai run --config runtime.yaml --duration-s 60
```

## API Surface

### FromConfig

`FromConfig` is class-level construction sugar.

```python
policy = Pi05.from_config(Pi05Config(chunk_size=50))
policy = Pi05.from_config({"chunk_size": 50})
policy = Pi05.from_config("pi05.yaml")
```

For ordinary classes, `from_config` delegates to generic instantiation.

```python
class FromConfig:
    @classmethod
    def from_config(cls, config: ConfigInput) -> Self:
        return instantiate_obj(config, target_cls=cls)
```

For orchestrators, `from_config` may bind subcomponents.

```python
class Trainer(FromConfig):
    @classmethod
    def from_config(cls, config: ConfigInput) -> "Trainer":
        cfg = TrainingConfig.model_validate(load_config(config))
        model = instantiate_component(cfg.model)
        data = instantiate_component(cfg.data)
        trainer = instantiate_component(cfg.trainer)
        trainer._configured_model = model
        trainer._configured_datamodule = data
        return trainer
```

`fit()` remains the public execution API:

```python
trainer.fit(model=model, datamodule=data)

# Or, when constructed from a full config:
Trainer.from_config("training.yaml").fit()
```

### ToConfig

`ToConfig` is opt-in serialization.

```python
class ToConfig:
    def to_config(self) -> Config | ComponentSpec:
        raise NotImplementedError
```

Objects decide what state is reproducible constructor config and what state is runtime-only.

```python
class Pi05(ToConfig):
    def to_config(self) -> Pi05Config:
        return Pi05Config(
            chunk_size=self.chunk_size,
            n_action_steps=self.n_action_steps,
        )
```

Component serialization can wrap object config with a target.

```python
spec = ComponentSpec.from_instance(policy)
```

Possible output:

```yaml
class_path: physicalai.policies.pi05.Pi05
init_args:
  chunk_size: 50
  n_action_steps: 50
```

### ConfigIO

`ConfigIO` can be a convenience mixin when a class supports both directions.

```python
class ConfigIO(FromConfig, ToConfig):
    pass
```

This should not make serialization automatic for every class. Inbound instantiation can be generic. Outbound serialization should be explicit.

## CLI Parity

The CLI should use the same config schema as the Python API.

```bash
physicalai run --config runtime.yaml
```

```python
PolicyRuntime.from_config("runtime.yaml").run()
```

These should construct the same runtime.

`physicalai` is the torch-free runtime package. Its CLI can depend on `jsonargparse`, but it must not depend on torch or Lightning.

```text
physicalai
  infer
  run
  serve
  inspect-manifest

physicalai-train or library distribution
  fit
  validate
  test
  predict
  export
```

Training commands can plug into the runtime CLI via entry points, but importing `physicalai` alone should not import training dependencies.

```toml
[project.entry-points."physicalai.cli.subcommands"]
run = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"

[project.entry-points."physicalai.cli.subcommands"]
fit = "physicalai.train.cli:register_fit"
```

## Package Boundaries

The generic configuration primitive belongs in the torch-free runtime package.

```text
physicalai/
  config/
    ComponentSpec
    instantiate_component
    FromConfig
    ToConfig
    ConfigIO

  inference/
    Manifest
    InferenceModel
    InferenceConfig

  runtime/
    PolicyRuntime
    RuntimeConfig

library/
  train/
    Trainer
    TrainingConfig

  export/
    manifest builders
```

The existing `physicalai.inference.manifest.ComponentSpec` can be moved or re-exported later.

Compatibility path:

```python
# physicalai.inference.manifest
from physicalai.config import ComponentSpec
```

Initial implementation can keep the class where it is and avoid churn. The important part is to treat it as the shared primitive.

## Design Rules

1. Use `ComponentSpec` for generic instantiable components.
2. Use typed `Config` classes for constructor arguments when validation or IDE support is useful.
3. Use workflow configs for workflow-level configuration.
4. Use manifests for exported package metadata.
5. Keep execution in orchestrators, not in config objects.
6. Keep torch and Lightning out of the `physicalai` runtime package.
7. Make inbound config generic; make outbound config explicit.
8. Put application payload translation in the application layer, but put reusable workflow configs and orchestration in the library/runtime layer.

## Migration Plan

### Phase 1: Stabilize ComponentSpec

- Treat existing `ComponentSpec` as the shared component descriptor.
- Add helpers if needed:

```python
ComponentSpec.from_config(target=Pi05, config=Pi05Config(...))
ComponentSpec.from_instance(policy)
```

- Align recursive instantiation between `instantiate_component` and `instantiate_obj_from_dict`.

### Phase 2: Add TrainingConfig

- Add a library-level `TrainingConfig`.
- Use `ComponentSpec` for `model`, `data`, and `trainer`.
- Keep callbacks under `trainer.init_args.callbacks`.
- Implement `Trainer.from_config(config)`.
- Implement `Trainer.to_config()` for round-trip serialization.

### Phase 3: Add Runtime/Inference Configs

- Add `InferenceConfig` only if manifest is not enough for authoring inference pipelines.
- Add `RuntimeConfig` around `PolicyRuntime` construction.
- Reuse the existing runtime design:

```python
runtime = PolicyRuntime.from_config("runtime.yaml")
runtime.run(duration_s=60)
```

### Phase 4: CLI

- Add a jsonargparse-based CLI in the torch-free `physicalai` package.
- Load the same YAML files accepted by Python APIs.
- Keep training commands in the training distribution or plugin entry points.

### Phase 5: Manifest/Workflow Config Convergence

- Keep workflow config and manifest separate initially.
- Reuse `ComponentSpec` in both.
- Converge schemas only where the same data is used for both authoring and package loading.

Example convergence target:

```python
model = InferenceModel.from_config("export/manifest.json")
model = InferenceModel.from_config("inference.yaml")
```

Both should go through the same component instantiation path.

## Open Questions

- Should `ComponentSpec` live in `physicalai.config` or `physicalai.specs`?
- Should `TrainingConfig.trainer` be a full `ComponentSpec` from the first version, or a typed `TrainerConfig` section temporarily?
- Should `Manifest` be accepted directly by `InferenceModel.from_config`, or normalized into `InferenceConfig` first?
- How much of `ComponentSpec` registry mode should be exposed in training configs?
- Should `Config` classes optionally declare a default target, or should `ComponentSpec.from_config(target=..., config=...)` remain explicit?

The conservative default is explicit targets, generic `ComponentSpec`, and domain workflow configs around orchestrators.
