# Policy Plugins: Registry, Bridge, and Native Packages

> Status: alternative design proposal, 2026-10-12. The code is illustrative pseudocode, not an implemented API. Read it alongside [Policy Plugin and Model Enablement Strategy](plugin-options.md); the last section compares the two proposals.

Paths are relative to the Studio repository root unless marked Runtime, which refers to [`openvinotoolkit/physicalai`](https://github.com/openvinotoolkit/physicalai).

## Summary

Studio needs a fast lane for external frameworks such as LeRobot or Xiaomi's, so a model can be trained in Studio and deployed with Runtime before anyone commits to a first-party port.

This proposal copies Studio's robot-catalog plugin pattern for policies, adds one thin bridge base class, and deploys plugin models as framework-native safetensors packages. Most of the machinery already exists:

- **Deploy.** The Torch export lane already deploys any `Policy` subclass. Runtime re-imports the class and runs it, so plugins never need their own Runtime adapter.
- **Discovery.** The robot catalog (`application/backend/src/robots/catalog/registry.py`) already has the plugin mechanics: an SDK dataclass, a `register(registry)` entry point, built-ins registered first, duplicate rejection, and attribution of each robot type to the distribution that registered it.

## Current State

What stands in the way today:

- **Policy identity is hardcoded.** The first-party XR0 integration touched about 20 files across both repositories. These include the model reimplementation, Runtime pre- and postprocessors, and policy name lists in `get_physicalai_policy_class()`, four backend modules under `application/backend/src/` (`api/policies.py`, `training/job.py`, `services/model_import_service.py`, `trainer/schemas.py`), and the UI's `policies.ts`. `TrainingJobSpec.policy_source` is `Literal["physicalai", "lerobot"]`.
- **The Torch lane uses pickle.** `ExportablePolicyMixin.to_torch()` writes the checkpoint with `torch.save`. `TorchAdapter.load()` in `library/src/physicalai/inference/adapters/pytorch.py` imports `policy.source.class_path` from the manifest and calls `load_from_checkpoint(..., weights_only=False)`. Runtime's `PolicySource` docstring describes that field as "Informational — the inference runtime does not import this".
- **The LeRobot wrapper is stateful at inference.** `TorchAdapter.predict()` calls `policy(observation)`. In eval mode, `LeRobotPolicy.forward()` returns one action from LeRobot's internal `select_action` queue, and Runtime's `InferenceModel.reset()` never reaches the adapter, so that queue carries over between episodes.

## Options

### Option A: Class Path Only

This works today, with no core changes.

```python
# acme_policies/policy.py: depends on physicalai-train
class AcmeVLA(ExportablePolicyMixin, Policy):
    def forward(self, batch: Observation): ...  # loss in train mode, chunk in eval
    def predict_action_chunk(self, batch: Observation): ...
    def training_step(self, batch, batch_idx): ...
    def configure_optimizers(self): ...
```

```bash
physicalai fit --model acme_policies.policy.AcmeVLA --data ...
physicalai export --policy acme_policies.policy.AcmeVLA --ckpt_path last.ckpt --backend torch --output_dir out
```

```python
InferenceModel("out")  # TorchAdapter imports class_path, then unpickles the checkpoint
```

This is fine for a one-off experiment. The model isn't in the Studio GUI, every plugin rewrites the Lightning glue, and deployment goes through pickle.

### Option B: Policy Registry Through Entry Points

A plugin declares one entry point and one `register` function:

```toml
# physicalai-xiaomi-policies/pyproject.toml
[project.entry-points."physicalai.policies"]
xiaomi = "physicalai_xiaomi_policies:register"
```

```python
# physicalai_xiaomi_policies/__init__.py
from physicalai.policies.registry import PolicyRegistry, PolicySpec


def register(registry: PolicyRegistry) -> None:
    registry.register(
        PolicySpec(
            id="xiaomi/r0",
            class_path="physicalai_xiaomi_policies.policy.R0Policy",  # imported only when used
            display_name="Xiaomi Robotics-0",
            description="Qwen3-VL backbone with a flow-matching action expert",
            license="Apache-2.0",
            min_vram_gb=60,
            hub_repos=("XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",),
        )
    )
```

Studio adds `library/src/physicalai/policies/registry.py`, mirroring the robot catalog registry:

```python
ENTRY_POINT_GROUP = "physicalai.policies"  # same style as physicalai.inference.adapters


@dataclass(frozen=True)
class PolicySpec:
    """Catalog facts Studio needs without importing the model."""

    id: str  # "<provider>/<name>": "physicalai/act", "lerobot/diffusion", "xiaomi/r0"
    class_path: str
    display_name: str
    description: str = ""
    license: str = ""
    min_vram_gb: float | None = None
    base_checkpoint: str | None = None
    hub_repos: tuple[str, ...] = ()


class PolicyRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, PolicySpec] = {}
        for spec in BUILTIN_SPECS:  # first-party and LeRobot use the same API as plugins
            self.register(spec)
        self._load_external_plugins()  # error-isolated and attributed, as in the robot catalog

    def register(self, spec: PolicySpec) -> None:
        if spec.id in self._specs:
            raise ValueError(f"Duplicate policy registration: {spec.id}")
        self._specs[spec.id] = spec

    def list(self) -> list[PolicySpec]:
        return list(self._specs.values())

    def get_class(self, policy_id: str) -> type[Policy]:
        return import_dotted_path(self._specs[policy_id].class_path)


@cache
def policy_registry() -> PolicyRegistry:
    return PolicyRegistry()
```

The registry replaces the hardcoded lists:

```python
# library: get_policy keeps its signature but drops the if-chain
def get_policy(policy_name: str, *, source: str = "physicalai", **kwargs) -> Policy:
    return policy_registry().get_class(f"{source}/{policy_name}")(**kwargs)
    # get_policy("diffusion", source="lerobot") still works


# backend: _POLICY_CLASSES, both _SUPPORTED_POLICIES, PRETRAINED_BASE_CHECKPOINTS and
# _HUGGINGFACE_REQUIREMENTS become registry lookups. PEFT and SnapFlow support become
# issubclass(cls, PeftPolicyMixin) checks, because the class already knows that.
@router.get("")
def list_policies() -> list[PolicyInfo]:
    return [PolicyInfo.from_spec(spec) for spec in policy_registry().list()]


# UI: the MODELS list in policies.ts is replaced by GET /api/policies
```

Plugins become discoverable and appear in the GUI, with one source of truth. They still write full Lightning modules, though, and still deploy through pickle.

### Option C (Recommended): Registry, Bridge, and Native Packages

Option C adds three pieces on top of Option B.

#### Bridge Base Class

A bridge base class owns the Lightning glue. It generalizes `LeRobotPolicy` and would live in a new `library/src/physicalai/policies/external.py`:

```python
class ExternalPolicy(ExportablePolicyMixin, Policy):
    """A policy whose model lives in another framework. Plugins implement five hooks."""

    POLICY_ID: ClassVar[str]  # must match the registered PolicySpec.id (like LeRobot's POLICY_NAME)

    def __init__(self, *, learning_rate: float = 1e-4, **options: Any) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.options = options  # framework-specific settings, passed through from YAML or the GUI
        self.save_hyperparameters()

    # The plugin contract
    @abstractmethod
    def build_model(self, features: DatasetFeatures) -> nn.Module: ...  # not configure_model: Lightning uses that name

    @abstractmethod
    def compute_loss(self, batch: Observation) -> tuple[Tensor, dict[str, Tensor]]: ...

    @abstractmethod
    def predict_action_chunk(self, batch: Observation) -> Tensor: ...  # (B, T, D)

    @abstractmethod
    def save_pretrained(self, path: Path) -> None: ...  # framework-native, safetensors, no pickle

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path: Path) -> Self: ...

    # Glue that plugins never write
    def setup(self, stage: str) -> None:
        if self.model is None:  # LeRobotPolicy.setup logic, generalized
            self.model = self.build_model(DatasetFeatures.from_datamodule(self.trainer.datamodule))

    def forward(self, batch: Observation):
        return self.compute_loss(batch) if self.training else self.predict_action_chunk(batch)

    def training_step(self, batch: Observation, batch_idx: int) -> Tensor:
        loss, parts = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log_dict({f"train/{k}": v for k, v in parts.items()})
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:  # override for parameter groups
        return torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=self.learning_rate)

    @staticmethod
    def get_supported_export_backends() -> list[str | ExportBackend]:
        return [ExportBackend.NATIVE]
```

#### A Plugin in About 40 Lines

In this example the `xiaomi_robotics` calls are placeholders for whatever the framework's real API is:

```python
from physicalai.data.lerobot import FormatConverter
from physicalai.policies.external import ExternalPolicy


class R0Policy(ExternalPolicy):
    POLICY_ID = "xiaomi/r0"

    def build_model(self, features):
        from xiaomi_robotics import R0ForAction  # placeholder; imported lazily

        return R0ForAction.from_pretrained(self.options.get("pretrained", "XiaomiRobotics/Xiaomi-Robotics-0-Pretrain"))

    def compute_loss(self, batch):
        out = self.model(**FormatConverter.to_lerobot_dict(batch))
        return out.loss, {}

    def predict_action_chunk(self, batch):
        return self.model.sample_actions(**FormatConverter.to_lerobot_dict(batch))

    def save_pretrained(self, path):
        self.model.save_pretrained(path)  # config.json + model.safetensors

    @classmethod
    def from_pretrained(cls, path):
        from xiaomi_robotics import R0ForAction

        policy = cls()
        policy.model = R0ForAction.from_pretrained(path)
        return policy
```

LeRobot would be the first adopter. `LeRobotPolicy` already has `save_pretrained()` and `from_pretrained()` (`config.json` plus `model.safetensors`) and `predict_action_chunk()`. Moving it onto the bridge proves the API and removes the stateful eval path.

#### Native Packages and One Generic Adapter

Every plugin exports the same package shape:

```text
exports/native/
├── manifest.json   {"policy": {"name": "r0", "source": {"policy_id": "xiaomi/r0"}},
│                    "model": {"artifacts": {"native": "model"}}}
└── model/          written by the plugin's save_pretrained(): config.json, model.safetensors
```

One adapter loads all of them:

```python
# library/src/physicalai/inference/adapters/native.py, registered lazily by the existing _register.py
@adapter_registry.register("native", extensions=())
class NativePolicyAdapter(RuntimeAdapter):
    def load(self, model_path: Path) -> None:
        manifest = Manifest.load(model_path.parent)
        policy_cls = policy_registry().get_class(manifest.policy.source.policy_id)  # registered ids only
        self._policy = policy_cls.from_pretrained(model_path).to(self.device).eval()

    @torch.no_grad()
    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        chunk = self._policy.predict_action_chunk(Observation.from_dict(inputs).to_torch(self.device))
        return {"action": chunk.float().cpu().numpy()}

    def reset(self) -> None:
        self._policy.reset()

    # input_names -> [] (consumes the whole Observation); output_names -> ["action"]
```

Resolving the class through the registry makes the registry an allowlist: a manifest can only name a registered policy, never an arbitrary `class_path`. That is what rule 4 of Runtime's `docs/development/security.md` asks for.

One generic adapter is preferred over one per framework because every per-framework adapter would redo loading and `Observation` conversion. With the bridge, framework specifics stay in the five hooks, and Runtime never needs to know about frameworks.

#### Runtime Changes

Runtime needs three small, generic changes, all useful beyond plugins:

```python
# Runtime: src/physicalai/inference/model.py
def _get_model_path(self) -> Path:
    if artifact := self.manifest.model.artifacts.get(self.backend):
        return resolve_contained(self.export_dir, artifact)  # reuse resolve_artifact's containment check
    ...  # existing extension lookup, unchanged


def reset(self) -> None:
    self.runner.reset()
    self.adapter.reset()  # new RuntimeAdapter.reset(), a no-op by default
    ...


# Runtime: src/physicalai/inference/manifest.py
class PolicySource(BaseModel):
    ...
    policy_id: str = ""  # older Runtimes ignore unknown fields
```

The first change also aligns the loader with Runtime's `docs/reference/manifest-schema.md`, which describes `artifacts` as a mapping to file paths. Today the loader reads only the key and finds the file by extension.

On the Studio side, `NATIVE` has to be added to `ExportBackend` and to the backend's two mirror enums in `application/backend/src/schemas/job.py` and `application/backend/src/schemas/hardware.py`. After that, the post-training export loop and the GUI's live inference pick it up with no further changes.

#### Graduating to First-Party

Moving a model to first-party is a new spec, not a rewrite of the integration:

```python
# Day 1: plugin, out of tree
PolicySpec(id="xiaomi/r0", class_path="physicalai_xiaomi_policies.policy.R0Policy", ...)

# Later: first-party, in tree. Both can coexist, so they can be compared in the GUI.
PolicySpec(id="physicalai/xr0", class_path="physicalai.policies.XR0", ...)
# The first-party class adds onnx and openvino through get_supported_export_backends()
```

The trade-off is that in the fast lane the robot runs PyTorch plus the framework. ONNX and OpenVINO come with the move to first-party.

## Comparison

| Aspect             | Option A: class path                            | Option B: registry                    | Option C: registry, bridge, native                                            |
| ------------------ | ----------------------------------------------- | ------------------------------------- | ----------------------------------------------------------------------------- |
| Plugin writes      | Full Lightning `Policy`                         | Full `Policy` and `register()`        | Five hooks and `register()`                                                   |
| Studio GUI         | No                                              | Yes                                   | Yes                                                                           |
| Deploy artifact    | Pickled `.pt`, class imported from the manifest | Same as A                             | Safetensors, class resolved through the registry                              |
| Inference contract | Whatever eval `forward()` returns               | Same as A                             | `predict_action_chunk()` and `reset()`                                        |
| Core changes       | None                                            | Registry; backend and UI read from it | Option B, plus bridge, `native` export and adapter, and three Runtime changes |

## Rollout

Option B is step 1 of Option C, so starting with B throws nothing away. Each step ships on its own:

1. **Registry.** First-party and LeRobot policies register as built-ins, and the backend and UI read from the registry. Behavior doesn't change, and the hardcoded lists go away.
2. **Bridge and native lane.** Add `ExternalPolicy`, the `native` export and adapter, and the Runtime changes, then port `LeRobotPolicy` onto the bridge as the proving case.
3. **First out-of-tree plugin.** Xiaomi or whichever framework is next.

## Open Decisions

- **Naming.** Use `provider/name` IDs and `physicalai.policies` as the entry-point group?
- **Trust.** Entry-point plugins already fall under Runtime's security model, where installing a package means trusting it, as with robot plugins. Should the GUI also label unverified plugins, reusing `TRUST_UNVERIFIED_PLUGIN` from Runtime's `packages/physicalai-lerobot-plugin`? The curated `application/backend/src/plugins/manifest.json` could gain a `policies` list next to `robots`, which would give GUI install and uninstall through the existing `PluginManager`.
- **Scope.** The bridge assumes PyTorch training. JAX models or TensorRT engines would be inference-only plugins through Runtime's existing `physicalai.inference.adapters` entry-point group.

## Relationship to the Plugin-Options Proposal

[Policy Plugin and Model Enablement Strategy](plugin-options.md) and this proposal agree on most of the direction:

- Discover policies through Python entry points instead of hardcoded lists.
- Keep `get_policy(name, source=...)` working during migration.
- Build one catalog that the backend and UI read from.
- Treat LeRobot as the reference integration, and escalate to first-party only when needed.
- Keep `get_supported_export_backends()` on the policy class.

They differ in five places:

| Topic              | plugin-options.md                                                                                 | This proposal                                                           | Trade-off                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Entry-point target | Maps a stable ID directly to a `Policy` class (`physicalai.policies.v1`)                          | Names a `register(registry)` callable (`physicalai.policies`)           | Direct mapping needs no registration API; a callback can register descriptors without importing classes, and matches the robot catalog and adapter registry                                   |
| Catalog metadata   | Attributes on the policy class                                                                    | A `PolicySpec` descriptor; behavioral capabilities stay on the class    | Class attributes cannot drift, but building the catalog imports every policy module; a descriptor lists without imports but can drift, mitigated by checking `POLICY_ID` when the class loads |
| Registry object    | No public object; `get_policy_class()` and `list_policies()` over a private cache                 | `PolicyRegistry` passed to plugins, plus a `policy_registry()` accessor | Smaller public surface versus an explicit registration API                                                                                                                                    |
| Shared base class  | None; framework code lives in each adapter class                                                  | `ExternalPolicy` with five hooks                                        | No new abstraction versus less glue per framework                                                                                                                                             |
| Deployment         | OpenVINO export through `ExportablePolicyMixin`; native artifacts kept for the original framework | `native` export with one generic adapter and three Runtime changes      | No Runtime changes, but a model cannot deploy until it exports, versus immediate deployment at the cost of PyTorch and the framework on the robot                                             |

IDs also differ in form (`lerobot.act` versus `lerobot/act`). Runtime's `policy.name` validation accepts dots but not slashes, which matters only if the ID is stored in that field; this proposal stores it in `policy.source.policy_id`.

[plugin-options.md](plugin-options.md) also covers support levels, qualification contracts, the `wrap(native)` convention, and a downstream patch policy, which this proposal does not address. Those compose with either discovery design. The native deployment lane depends only on resolving a policy ID to a class, so it could also be adopted on top of the class-centric discovery in [plugin-options.md](plugin-options.md).

## References

- `library/src/physicalai/policies/__init__.py`: `get_policy()` and `get_physicalai_policy_class()`
- `library/src/physicalai/policies/lerobot/policy.py`: `LeRobotPolicy`
- `library/src/physicalai/export/mixin_policy.py`: `ExportablePolicyMixin`
- `library/src/physicalai/inference/adapters/pytorch.py`: `TorchAdapter`
- `application/backend/src/robots/catalog/registry.py`: robot catalog registry
- `application/backend/src/training/job.py`: `TrainingJobSpec`, `build_policy()`, and post-training export
- Runtime `src/physicalai/inference/model.py`: `InferenceModel`
- Runtime `src/physicalai/inference/adapters/registry.py`: `RuntimeAdapterRegistry`
- Runtime `src/physicalai/inference/manifest.py`: `Manifest` and `PolicySource`
- Runtime `docs/development/security.md`: dynamic-construction trust boundaries (rule 4)
