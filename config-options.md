# Configuration Options For Alignment

## Context

Goal: one config schema serves Python, CLI, API, GUI, checkpoints, export.

Options under discussion:

- constructor shape
- config dataclass shape
- UI grouping location

Recommended direction:

- flat kwargs constructors
- `FromConfig` as adapter into constructors
- config shape + UI grouping remain open

```python
policy = Pi05(chunk_size=100, optimizer_lr=1e-5)
policy = Pi05.from_config(Pi05Config(chunk_size=100))
```

## Key Risks

### `config=` Constructor

```python
Pi05(config=Pi05Config(...))
```

- `Lightning` hparams sees config as one arg, not flat fields.
- Checkpoint reload needs config-object reconstruction.
- Hub-loaded models need config schema translation regardless.
- Migration cost is high across policies, checkpoints, export.

### Why Flat Kwargs Is The Current Recommendation

- Keeps `Lightning` hparams aligned with ctor args.
- Keeps checkpoint reload direct.
- Keeps PhysicalAI-to-upstream config mapping explicit.
- Keeps constructor semantics unchanged for model code.
- Keeps CLI flat and unsurprising.

### Flat Constructor With Large Signatures

```python
def __init__(
    self,
    chunk_size=50,
    n_action_steps=50,
    optimizer_lr=2.5e-5,
    scheduler_warmup_steps=1000,
    freeze_vision_encoder=True,
    train_expert_only=False,
    ...
):
    ...
```

- Strengths: explicit, `Lightning`-safe, direct upstream mapping.
- Cost: hard to group for UI, long signatures, field ownership less visible.

### UI Needs

```json
{"groups": [{"id": "io", "fields": ["chunk_size", "n_action_steps"]}]}
```

Needs:

- types
- defaults
- descriptions
- validation
- grouping
- visibility
- stable payload

### Metadata Ownership

Question: where does grouping live?

- library metadata
- application overlay
- nested dataclass structure

## Options

Recommended baseline for options A-D:

```python
class Pi05(FromConfig):
    def __init__(self, chunk_size=50, n_action_steps=50, optimizer_lr=2.5e-5):
        self.config = Pi05Config(...)
        self.save_hyperparameters(ignore=["config"])
        self.hparams["config"] = self.config.to_dict()
```

Option E changes the schema layer, not the constructor recommendation.

### Option A: Flat Dataclass

```python
@dataclass(frozen=True)
class Pi05Config:
    chunk_size: int = 50
    n_action_steps: int = 50
    optimizer_lr: float = 2.5e-5
    freeze_vision_encoder: bool = True
```

Pros:

- lowest migration
- simplest
- hparams flat
- CLI short

Cons:

- no grouping
- UI must map externally or via metadata

### Option B: Flat Dataclass + App-Owned Grouping

```python
# Library: clean, no metadata

# Application:
PI05_FIELD_GROUPS = {
    "io": ["chunk_size", "n_action_steps"],
    "optimizer": ["optimizer_lr"],
    "fine_tuning": ["freeze_vision_encoder"],
}
```

Pros:

- library stays clean
- Studio controls grouping fully

Cons:

- drift risk: new fields need app updates
- needs CI guard

```python
def test_all_fields_grouped():
    assert {f.name for f in fields(Pi05Config)} <= set(chain.from_iterable(PI05_FIELD_GROUPS.values()))
```

### Option C: Flat Dataclass + Library Group Metadata

```python
@dataclass(frozen=True)
class Pi05Config:
    chunk_size: int = field(default=50, metadata={"group": "io"})
    n_action_steps: int = field(default=50, metadata={"group": "io"})
    optimizer_lr: float = field(default=2.5e-5, metadata={"group": "optimizer"})
    freeze_vision_encoder: bool = field(default=True, metadata={"group": "fine_tuning"})
```

```python
# Application overlay: presentation only
PI05_UI_OVERLAY = {
    "io": {"label": "Input / Output", "order": 10},
    "optimizer": {"label": "Optimizer", "order": 90, "advanced": True},
    "chunk_size": {"label": "Chunk Size", "widget": "number"},
}
```

Pros:

- grouping co-located with field
- low drift
- Studio still owns presentation

Cons:

- adds metadata to library
- boundary discipline required

Boundary:

- Library owns: group id, description, validation hint, deprecated flag
- Studio owns: label, order, widget, advanced/hidden, product copy

### Option D: Nested Dataclass

```python
@dataclass(frozen=True)
class Pi05IOConfig:
    chunk_size: int = 50
    n_action_steps: int = 50

@dataclass(frozen=True)
class Pi05OptimizerConfig:
    lr: float = 2.5e-5

@dataclass(frozen=True)
class Pi05Config:
    io: Pi05IOConfig = field(default_factory=Pi05IOConfig)
    optimizer: Pi05OptimizerConfig = field(default_factory=Pi05OptimizerConfig)
```

```python
# Constructor stays flat; __init__ maps kwargs -> nested config
def __init__(self, chunk_size=50, n_action_steps=50, optimizer_lr=2.5e-5):
    self.config = Pi05Config(
        io=Pi05IOConfig(chunk_size=chunk_size, n_action_steps=n_action_steps),
        optimizer=Pi05OptimizerConfig(lr=optimizer_lr),
    )

# from_config must flatten nested -> kwargs
@classmethod
def from_config(cls, config):
    return cls(chunk_size=config.io.chunk_size, ...)
```

Pros:

- structural grouping
- clean nested payload
- local validation per group

Cons:

- field aliases (`optimizer_lr` vs `optimizer.lr`)
- flatten/unflatten adapters
- every new field needs group placement + mapping maintenance

```json
{
  "policy_config": {
    "io": {"chunk_size": 100},
    "optimizer": {"lr": 0.00001}
  }
}
```

### Option E: Pydantic API Models

```python
class Pi05ConfigModel(BaseModel):
    chunk_size: int = Field(default=50, gt=0, json_schema_extra={"group": "io"})
    optimizer_lr: float = Field(default=2.5e-5, gt=0, json_schema_extra={"group": "optimizer"})

class TrainJobPayload(BaseModel):
    policy: Literal["pi05", "act", "smolvla"]
    policy_config: Pi05ConfigModel | ACTConfigModel | SmolVLAConfigModel
```

Pros:

- FastAPI/OpenAPI native
- strong validation
- schema generation built in

Cons:

- duplicates dataclass unless generated
- needs Pydantic -> kwargs mapper
- dependency placement decision needed

Placement variants:

- E1: Pydantic in library — single schema, stronger coupling
- E2: Pydantic in application — library stays dataclass, mapper needed
- E3: generated from dataclass — less duplication, more tooling

## CLI Impact

YAML is optional. CLI flags instantiate directly. Same for all options.

```bash
physicalai fit --model Pi05 --model.chunk_size 100 --model.optimizer_lr 0.00001
```

```yaml
model:
  class_path: Pi05
  init_args:
    chunk_size: 100
    optimizer_lr: 0.00001
```

Constructor is flat in all options, so CLI stays flat regardless of config shape.

## API Payload

### Flat: options A, B, C, E

```json
{"policy": "pi05", "policy_config": {"chunk_size": 100, "optimizer_lr": 0.00001}}
```

```python
policy = Pi05(**payload.policy_config)
```

### Grouped: option D, or C/E with app mapping

```json
{"policy": "pi05", "policy_config": {
  "io": {"chunk_size": 100},
  "optimizer": {"optimizer_lr": 0.00001}
}}
```

```python
policy = Pi05(**flatten_grouped_payload(payload.policy_config))
```

## GUI Schema Contract

```python
class ConfigFieldSchema(BaseModel):
    path: str
    type: str
    default: Any
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

```python
@app.get("/policies/{policy}/config-schema")
def get_schema(policy: str) -> PolicyConfigSchema:
    return build_ui_schema(POLICY_REGISTRY[policy].config_cls, POLICY_REGISTRY[policy].ui_overlay)
```

## Decision Matrix

| Option | Config Shape | UI Group Source | Migration | Drift Risk | OpenAPI | Notes |
| ------ | ------------ | --------------- | --------- | ---------- | ------- | ----- |
| A | flat | none/external | lowest | high | weak | simplest |
| B | flat | application | low | high (testable) | medium | library clean |
| C | flat | lib metadata + app overlay | low-medium | low | medium | pragmatic |
| D | nested | structural | medium-high | low | medium | alias cost |
| E | flat/nested | Pydantic schema | medium | depends | strong | strongest schema |

| Concern | A | B | C | D | E |
| ------- | - | - | - | - | - |
| Model/upstream mapping | strong | strong | strong | medium | strong (adapter) |
| Lightning hparams | strong | strong | strong | strong | strong (adapter) |
| UI grouping | weak | medium | strong | strong | strong |
| Library purity | strong | strong | medium | medium | weak/medium |

## Recommendation

Recommended:

- constructor = flat kwargs
- `FromConfig` = adapter into kwargs

Still open:

- A: flat dataclass
- B: flat dataclass + app grouping
- C: flat dataclass + library semantic metadata
- D: nested dataclass
- E: Pydantic schema layer

Why:

- strongest checkpoint/hparams path
- lowest disruption to model code
- clearest upstream config mapping
- keeps CLI/API constructor semantics stable

## Open Questions

1. Is minimal semantic metadata in library acceptable?
2. Flat or grouped API payload?
3. Pydantic in library, application, or generated adapter?
4. Are nested dataclasses worth the alias/mapping cost?
5. Who owns field grouping when a new parameter is added?
