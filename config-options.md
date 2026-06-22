# Configuration Options

## Scope

```text
Subject:
  object construction
  config schema
  CLI/API/GUI shape

Examples:
  Policy
  DataModule
  Trainer
  Runtime
  callbacks/presets
```

## Option A: Keep Current Shape

```python
class Pi05:
    def __init__(
        self,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        optimizer_lr: float = 2.5e-5,
        freeze_vision_encoder: bool = True,
        compile_model: bool = False,
        dataset_stats: dict | None = None,
    ) -> None:
        self.config = Pi05Config(
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            optimizer_lr=optimizer_lr,
            freeze_vision_encoder=freeze_vision_encoder,
            compile_model=compile_model,
        )
```

```python
policy = Pi05(chunk_size=100, optimizer_lr=1e-5)
```

```python
# Config dataclass exists, but constructor remains flat.
config = policy.config
```

### Pros

```text
simple Python call
explicit constructor signature
jsonargparse introspects directly
short CLI paths
low migration cost
matches current code
```

```bash
physicalai fit \
  --model.class_path Pi05 \
  --model.init_args.chunk_size 100 \
  --model.init_args.optimizer_lr 0.00001
```

### Cons

```text
large signatures for policies
weak semantic grouping
harder GUI form sections
harder OpenAPI/Pydantic mapping
cross-field validation scattered
config object is output, not input
REST payload is flat
```

```json
{
  "policy": "pi05",
  "policy_config": {
    "chunk_size": 100,
    "n_action_steps": 50,
    "optimizer_lr": 0.00001,
    "freeze_vision_encoder": true
  }
}
```

### Optional FromConfig Adapter

```python
@dataclass(frozen=True)
class Pi05Config:
    chunk_size: int = 50
    optimizer_lr: float = 2.5e-5


class Pi05(FromConfig):
    def __init__(self, chunk_size: int = 50, optimizer_lr: float = 2.5e-5) -> None:
        self.config = Pi05Config(
            chunk_size=chunk_size,
            optimizer_lr=optimizer_lr,
        )
```

```python
# Classmethod adapter unwraps config into flat kwargs.
policy = Pi05.from_config(Pi05Config(chunk_size=100))

# Constructor remains flat.
policy = Pi05(chunk_size=100)
```

### Pros

```text
keeps current constructor
adds typed config input path
lower migration cost
jsonargparse can still inspect flat signature
```

### Cons

```text
two construction paths
config object is adapter input, not constructor API
large signatures remain
GUI grouping still unresolved
not uniformly implemented today
```

## Option B: Flat Config Dataclass As Constructor Input

```python
@dataclass(frozen=True)
class Pi05Config:
    chunk_size: int = 50
    n_action_steps: int = 50
    optimizer_lr: float = 2.5e-5
    freeze_vision_encoder: bool = True

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")


class Pi05:
    def __init__(self, config: Pi05Config = Pi05Config()) -> None:
        self.config = config
```

```python
policy = Pi05(config=Pi05Config(chunk_size=100))
```

```python
# Constructor no longer exposes each config field directly.
policy = Pi05(chunk_size=100)  # not canonical
```

### Pros

```text
typed config object
small constructor
config is input and saved state
easy dataclass/Pydantic adapter
validation can live in config
```

### Cons

```text
still flat
GUI grouping needs metadata/overlay
less convenient than Pi05(chunk_size=100)
requires migration from flat kwargs
```

```json
{
  "policy": "pi05",
  "policy_config": {
    "chunk_size": 100,
    "optimizer_lr": 0.00001
  }
}
```

## Option C: Flat Config Dataclass + UI Group Metadata

```python
@dataclass(frozen=True)
class Pi05Config:
    chunk_size: int = field(
        default=50,
        metadata={"ui": {"group": "io", "label": "Chunk Size", "order": 10}},
    )
    n_action_steps: int = field(
        default=50,
        metadata={"ui": {"group": "io", "label": "Action Steps", "order": 20}},
    )
    optimizer_lr: float = field(
        default=2.5e-5,
        metadata={"ui": {"group": "optimizer", "label": "Learning Rate"}},
    )
```

```python
schema = build_ui_schema(Pi05Config)
```

```json
{
  "groups": [
    {"id": "io", "fields": ["chunk_size", "n_action_steps"]},
    {"id": "optimizer", "fields": ["optimizer_lr"]}
  ]
}
```

### Pros

```text
minimal payload migration
GUI can group fields
flat checkpoint compatibility easier
typed config object
```

### Cons

```text
grouping is presentation-only
semantic config remains flat
metadata may drift into UI policy
field paths differ from UI groups
```

```json
{
  "policy_config": {
    "chunk_size": 100,
    "optimizer_lr": 0.00001
  }
}
```

## Option D: Nested Config Dataclasses

```python
@dataclass(frozen=True)
class Pi05IOConfig:
    chunk_size: int = 50
    n_action_steps: int = 50

    def __post_init__(self) -> None:
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size")


@dataclass(frozen=True)
class Pi05OptimizerConfig:
    lr: float = 2.5e-5
    weight_decay: float = 1e-10


@dataclass(frozen=True)
class Pi05FineTuningConfig:
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False


@dataclass(frozen=True)
class Pi05Config:
    io: Pi05IOConfig = field(default_factory=Pi05IOConfig)
    optimizer: Pi05OptimizerConfig = field(default_factory=Pi05OptimizerConfig)
    fine_tuning: Pi05FineTuningConfig = field(default_factory=Pi05FineTuningConfig)


class Pi05:
    def __init__(self, config: Pi05Config = Pi05Config()) -> None:
        self.config = config
```

```python
policy = Pi05(
    config=Pi05Config(
        io=Pi05IOConfig(chunk_size=100),
        optimizer=Pi05OptimizerConfig(lr=1e-5),
    )
)
```

### Pros

```text
real semantic grouping
clean GUI/API payload
local validation per group
small constructors
config is public contract
maps to docs sections
```

### Cons

```text
larger migration
more public config classes
longer CLI paths
checkpoint compatibility needs care
group boundaries become API
```

```json
{
  "policy": "pi05",
  "policy_config": {
    "io": {
      "chunk_size": 100,
      "n_action_steps": 50
    },
    "optimizer": {
      "lr": 0.00001
    }
  }
}
```

## Option E: Pydantic Config Models

```python
class Pi05IOConfig(BaseModel):
    chunk_size: int = Field(default=50, gt=0, description="Action chunk size")
    n_action_steps: int = Field(default=50, gt=0)


class Pi05OptimizerConfig(BaseModel):
    lr: float = Field(default=2.5e-5, gt=0)


class Pi05Config(BaseModel):
    io: Pi05IOConfig = Field(default_factory=Pi05IOConfig)
    optimizer: Pi05OptimizerConfig = Field(default_factory=Pi05OptimizerConfig)


class Pi05:
    def __init__(self, config: Pi05Config = Pi05Config()) -> None:
        self.config = config
```

```python
schema = Pi05Config.model_json_schema()
```

### Pros

```text
strong validation
FastAPI/OpenAPI native
JSON schema native
GUI schema easier
typed API payloads
```

### Cons

```text
adds Pydantic to package boundary
different serialization model
checkpoint compatibility work
may be heavier for runtime package
```

## Constructor Rule Options

### Rule 1: Always Flat Kwargs

```python
Pi05(chunk_size=100, optimizer_lr=1e-5)
Resize(width=224, height=224)
```

```text
simple until signatures grow
weak GUI/API shape
```

### Rule 2: Always Config Objects

```python
Pi05(config=Pi05Config(...))
Resize(config=ResizeConfig(...))
```

```text
uniform
more config classes
overkill for small helpers
```

### Rule 3: Boundary-Based

```python
def use_config_object(cls) -> bool:
    return (
        cls.is_user_configurable
        or cls.is_saved_in_checkpoint
        or cls.is_exported_or_reloaded
        or cls.param_count >= 8
        or cls.has_semantic_groups
        or cls.has_cross_field_validation
        or cls.needs_api_schema
    )
```

```python
# kwargs
Resize(width=224, height=224)
Normalize(eps=1e-6)

# config object
Pi05(config=Pi05Config(...))
ACT(config=ACTConfig(...))
LeRobotDataModule(config=LeRobotDataModuleConfig(...))
PolicyRuntime(config=RuntimeConfig(...))
```

## Examplary Group Vocabulary

```text
io              input/output shape, horizons, chunking
architecture    model dimensions, layers, heads, backbones
preprocessing   resize, normalization, tokenization
fine_tuning     freeze flags, trainable subsets
optimizer       optimizer and scheduler knobs
inference       action selection/generation behavior
export          export-time knobs
runtime         robot/run-loop knobs
```

```python
@dataclass(frozen=True)
class ACTConfig:
    io: ACTIOConfig
    vision: ACTVisionConfig
    transformer: ACTTransformerConfig
    vae: ACTVAEConfig
    optimizer: ACTOptimizerConfig
    inference: ACTInferenceConfig
```

```python
@dataclass(frozen=True)
class Pi05Config:
    io: Pi05IOConfig
    backbone: Pi05BackboneConfig
    preprocessing: Pi05PreprocessingConfig
    flow_matching: Pi05FlowMatchingConfig
    fine_tuning: Pi05FineTuningConfig
    optimizer: Pi05OptimizerConfig
```

```text
Do not force every model family to have every group.
Use common names where natural.
Allow family-specific groups.
```

## Workflow Config Separation

```python
@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int = 10_000
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"


@dataclass(frozen=True)
class LeRobotDataModuleConfig:
    repo_id: str
    batch_size: int = 32
    num_workers: int = 4


@dataclass(frozen=True)
class TrainingConfig:
    trainer: TrainerConfig
    model: Pi05Config | ACTConfig | SmolVLAConfig
    data: LeRobotDataModuleConfig
```

```text
TrainingConfig contains PolicyConfig.
PolicyConfig should not contain TrainerConfig.
```

```python
config = TrainingConfig(
    trainer=TrainerConfig(max_steps=10_000),
    model=Pi05Config(io=Pi05IOConfig(chunk_size=100)),
    data=LeRobotDataModuleConfig(repo_id="physical-ai/example"),
)

Trainer.from_config(config).fit()
```

## Polymorphic Slot Shapes

### Python Typed Aggregate

```python
TrainingConfig(
    model=Pi05Config(...),
    data=LeRobotDataModuleConfig(...),
)
```

```python
def build_model(config):
    match config:
        case Pi05Config():
            return Pi05(config=config)
        case ACTConfig():
            return ACT(config=config)
```

### jsonargparse Shape

```python
parser.add_subclass_arguments(Policy, "model")
```

```yaml
model:
  class_path: Pi05
  init_args:
    config:
      io:
        chunk_size: 100
```

### API Shape

```json
{
  "policy": "pi05",
  "policy_config": {
    "io": {"chunk_size": 100}
  }
}
```

```python
config_cls = POLICY_REGISTRY[payload.policy].config_cls
model_config = config_cls.from_dict(payload.policy_config)
```

## Callback Surface

### Python-Only Ergonomics

```python
TrainerConfig(
    callbacks=[
        ModelCheckpoint(save_top_k=3),
        EarlyStopping(monitor="val/loss"),
    ]
)
```

```text
works in Python
not ideal for API/GUI/JSON/OpenAPI
live objects are not portable config
```

### Serializable Callback Config

```python
@dataclass(frozen=True)
class CheckpointingConfig:
    enabled: bool = True
    save_top_k: int = 3
    monitor: str = "val/loss"


@dataclass(frozen=True)
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    monitor: str = "val/loss"


@dataclass(frozen=True)
class CallbackPresetConfig:
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
```

```python
def build_callbacks(config: CallbackPresetConfig) -> list[Callback]:
    callbacks = []
    if config.checkpointing.enabled:
        callbacks.append(ModelCheckpoint(...))
    if config.early_stopping.enabled:
        callbacks.append(EarlyStopping(...))
    return callbacks
```

## jsonargparse CLI Impact

### No YAML Required

```text
CLI flags -> jsonargparse -> objects
YAML file -> jsonargparse -> objects
dict      -> jsonargparse -> objects
dataclass -> jsonargparse -> objects
Pydantic  -> jsonargparse -> objects
```

### Flat Kwarg Constructor

```python
class Pi05:
    def __init__(self, chunk_size: int = 50, optimizer_lr: float = 2.5e-5): ...
```

```bash
physicalai fit \
  --model.class_path Pi05 \
  --model.init_args.chunk_size 100 \
  --model.init_args.optimizer_lr 0.00001
```

### Config Constructor

```python
class Pi05:
    def __init__(self, config: Pi05Config = Pi05Config()): ...
```

```bash
physicalai fit \
  --model.class_path Pi05 \
  --model.init_args.config.io.chunk_size 100 \
  --model.init_args.config.optimizer.lr 0.00001
```

### YAML Equivalent

```yaml
model:
  class_path: Pi05
  init_args:
    config:
      io:
        chunk_size: 100
      optimizer:
        lr: 0.00001
```

### Optional CLI Sugar

```bash
physicalai fit \
  --policy pi05 \
  --policy.io.chunk_size 100 \
  --policy.optimizer.lr 0.00001
```

```python
def normalize_cli(args):
    return {
        "model": {
            "class_path": POLICY_REGISTRY[args.policy].class_path,
            "init_args": {"config": args.policy_config},
        }
    }
```

## API Impact

### Flat Payload

```json
{
  "policy": "pi05",
  "policy_config": {
    "chunk_size": 100,
    "optimizer_lr": 0.00001
  }
}
```

### Nested Payload

```json
{
  "policy": "pi05",
  "policy_config": {
    "io": {"chunk_size": 100},
    "optimizer": {"lr": 0.00001}
  }
}
```

### Backend Mapping

```python
def build_policy_config(payload: TrainJobPayload):
    config_cls = POLICY_REGISTRY[payload.policy].config_cls
    return config_cls.from_dict(payload.policy_config)


def build_training_config(payload: TrainJobPayload) -> TrainingConfig:
    return TrainingConfig(
        trainer=TrainerConfig(...),
        model=build_policy_config(payload),
        data=LeRobotDataModuleConfig(...),
    )
```

## GUI Impact

### Base Schema Contract

```python
class ConfigFieldSchema(BaseModel):
    path: str
    type: str
    default: Any
    label: str | None = None
    description: str | None = None
    validation: dict[str, Any] = {}
    advanced: bool = False


class ConfigGroupSchema(BaseModel):
    id: str
    label: str
    order: int = 100
    fields: list[ConfigFieldSchema]


class PolicyConfigSchema(BaseModel):
    policy: str
    display_name: str
    groups: list[ConfigGroupSchema]
```

### Endpoint Shape

```python
@app.get("/policies")
def list_policies(): ...


@app.get("/policies/{policy}/config-schema")
def get_config_schema(policy: str) -> PolicyConfigSchema: ...


@app.get("/policies/{policy}/default-config")
def get_default_config(policy: str) -> dict: ...
```

### UI Rendering

```tsx
const schema = await fetchPolicyConfigSchema(policy);

for (const group of schema.groups) {
  renderSection(group.label, group.fields);
}
```

## Metadata Location Options

### Library-Owned Semantic Metadata

```python
lr: float = field(
    default=2.5e-5,
    metadata={
        "description": "Optimizer learning rate.",
        "validation": {"gt": 0.0},
        "units": None,
    },
)
```

```text
field meaning
type/default
constraints
units
deprecation
```

### Studio-Owned Presentation Overlay

```python
PI05_UI_OVERLAY = PolicyUIOverlay(
    groups={
        "io": GroupOverlay(label="Input / Output", order=10),
        "optimizer": GroupOverlay(label="Optimizer", order=90, advanced=True),
    },
    fields={
        "io.chunk_size": FieldOverlay(label="Chunk Size", widget="number"),
        "optimizer.lr": FieldOverlay(label="Learning Rate", widget="scientific-number"),
    },
)
```

```text
labels/order
advanced/basic
widgets
visibility
product copy
presets
```

### Typed Metadata Helpers

```python
class GroupId(StrEnum):
    IO = "io"
    OPTIMIZER = "optimizer"
    INFERENCE = "inference"


@dataclass(frozen=True)
class Validation:
    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None


@dataclass(frozen=True)
class FieldInfo:
    description: str | None = None
    validation: Validation = Validation()


@dataclass(frozen=True)
class UIHint:
    label: str | None = None
    order: int = 100
    advanced: bool = False
```

```python
def config_field(*, default, info: FieldInfo, ui: UIHint = UIHint()):
    return field(default=default, metadata={"info": info, "ui": ui})
```

```python
lr: float = config_field(
    default=2.5e-5,
    info=FieldInfo(description="Optimizer learning rate.", validation=Validation(gt=0)),
    ui=UIHint(label="Learning Rate", order=10),
)
```

## Migration Compatibility

```python
class Pi05:
    def __init__(
        self,
        config: Pi05Config | None = None,
        *,
        chunk_size: int | None = None,
        optimizer_lr: float | None = None,
    ) -> None:
        if config is not None and (chunk_size is not None or optimizer_lr is not None):
            raise ValueError("Pass either config or flat kwargs, not both")

        if config is None:
            config = Pi05Config.from_legacy_kwargs(
                chunk_size=chunk_size,
                optimizer_lr=optimizer_lr,
            )

        self.config = config
```

```text
flat kwargs as temporary compatibility
config object as canonical future path
```

## Decision Matrix

| Option | Python UX | GUI UX | API/OpenAPI | CLI | Migration | Long-Term Shape |
| ------ | --------- | ------ | ----------- | --- | --------- | --------------- |
| A: Current flat kwargs | best for small | weak | weak | short flags | easiest | current |
| B: Flat config constructor | good | medium | medium | medium | medium | ok |
| C: Flat config + metadata | good | good | medium | medium | medium | transitional |
| D: Nested dataclasses | medium | good | good | longer flags | harder | strong |
| E: Pydantic models | medium | strong | strong | longer flags | harder | strong if dependency accepted |
