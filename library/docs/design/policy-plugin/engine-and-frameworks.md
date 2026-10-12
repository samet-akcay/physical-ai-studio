# Policy Plugins: Engine and Native Frameworks

> Status: design proposal, 2026-10-12. The code is illustrative pseudocode, not an implemented API. Read it alongside [Policy Plugin and Model Enablement Strategy](plugin-options.md) and [Policy Plugins: Registry, Bridge, and Native Packages](registry-and-bridge.md); the last section compares the three proposals.

Paths are relative to the Studio repository root unless marked Runtime, which refers to [`openvinotoolkit/physicalai`](https://github.com/openvinotoolkit/physicalai). LeRobot references are to the `v0.6.0` tag, the version `physicalai-train` pins.

## Summary

Physical AI should train, package, and deploy policies from other frameworks (LeRobot first, then others such as Xiaomi's or OpenPI) through its Python API, its CLI, and the Studio GUI, without porting them first. Someone who knows LeRobot should be able to use LeRobot exactly as they already do.

This proposal moves the integration seam from the model to the job:

- **The framework trains itself.** A framework plugin runs the framework's own training entry point (for LeRobot, the function behind `lerobot-train`) instead of placing the framework's model inside a Lightning loop.
- **Files are the contract.** The input is a LeRobot-format dataset, which Studio already records. The output is a Runtime policy package: `manifest.json` plus the framework's native model folder.
- **An `Engine` orchestrates.** `physicalai.engine.Engine` routes each request to a framework (`physicalai`, `lerobot`, or a third-party one) and adds what frameworks lack: progress and cancellation hooks, packaging, and Studio integration. It never defines a framework's training settings.
- **It lives in the library.** The Engine, the framework contract, and the registry ship in `physicalai-train`. Plugins are ordinary pip packages discovered through entry points. The Studio backend and GUI only consume the library API.

## Why Not Wrap the Training Loop

The current `LeRobotPolicy` puts LeRobot's model inside a Lightning module. That makes the model match LeRobot, but the result of training also depends on the training script: the learning-rate schedule, gradient clipping, the frames each sample contains, normalization statistics, precision, and checkpointing. Comparing `src/lerobot/scripts/lerobot_train.py` at `v0.6.0` with Studio's path (`library/src/physicalai/policies/lerobot/policy.py`, `library/src/physicalai/train/utils.py`, and `application/backend/src/training/job.py`) shows the differences below. They were found by reading the code and have not yet been reproduced in a run.

| `lerobot-train` (v0.6.0 defaults)                                                                                   | Studio today                                                                             | Effect                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Warmup and cosine LR schedule from the policy preset (every named wrapper except ACT)                               | Not built; constant LR                                                                   | Different optimization, especially for VLA fine-tuning                                                                                                              |
| Gradient clipping from the optimizer preset (10.0 for ACT, Diffusion, and SmolVLA; 1.0 for the pi family and GR00T) | None in production                                                                       | Updates are never clipped                                                                                                                                           |
| Action and observation windows from the config's `*_delta_indices`                                                  | `_get_delta_indices()` approximates them as `range(n_action_steps)`                      | Diffusion gets 32 actions from t instead of 64 from t-1 and fails `assert horizon == self.config.horizon`; ACT and others break when `n_action_steps != chunk_size` |
| `EpisodeAwareSampler` with `drop_n_last_frames` (7 for Diffusion)                                                   | Plain shuffle                                                                            | More samples padded past the end of the episode                                                                                                                     |
| Camera mean and std replaced with ImageNet statistics (`dataset.use_imagenet_stats=True`)                           | Dataset statistics                                                                       | Different image normalization for ACT and Diffusion                                                                                                                 |
| fp32 unless the policy's `dtype` asks for bf16 or fp16                                                              | GUI default `bf16-mixed`                                                                 | Different numerics                                                                                                                                                  |
| Fine-tuning refits normalizer statistics and output features to the new dataset; `rename_map` maps camera keys      | Pretrained processors and config are kept as they are                                    | Wrong normalization when fine-tuning; XR0 works around the same issue with `_rebuild_preprocessors()`                                                               |
| Held-out loss through `policy.forward()` (off by default; trains on all episodes)                                   | `validation_step()` only accepts gym environments, and the GUI holds out 10% of episodes | The first validation calls `env.reset()` on an `Observation` and fails                                                                                              |
| `policy.update()` after each optimizer step                                                                         | Never called                                                                             | Only TDMPC's EMA target network depends on it                                                                                                                       |
| 100k steps; keeps the last checkpoint                                                                               | 5 epochs; keeps the best `val/loss` checkpoint                                           | Different budget and checkpoint choice                                                                                                                              |
| `--peft`, per-sample loss weighting, language-annotated datasets                                                    | Not exposed                                                                              | Optional features are missing                                                                                                                                       |

Inference has similar gaps when a wrapped model runs through Runtime. `TorchAdapter.predict()` calls the policy's eval `forward()`, which returns one action from LeRobot's internal queue. `InferenceModel.reset()` never reaches the policy, so queues and observation history leak across episodes. LeRobot's real-time chunking arguments (`inference_delay` and `prev_chunk_left_over`) cannot be passed through.

`library/tests/integration/test_lerobot_wrapper_equivalence.py` proves that the wrapper reproduces LeRobot's loss, gradients, and optimizer step. It cannot catch the differences above. The native side replays the batches Studio produced, neither side uses a scheduler, gradient clipping is passed to the test trainer by hand, and the runs last a few steps (in fp32 for the non-VLA policies). The dataset is also built through `get_delta_timestamps_from_policy()`, with its default `fps=10` against a 50 fps dataset, instead of the production path.

Lightning is not the limitation. It can express all of this, and XR0 and RLDX-1 already configure step-based schedulers and gradient clipping. But each framework's recipe would have to be reimplemented and proven equivalent, and each framework release could change it again. Running the framework's own trainer makes training equivalent by construction.

## Design

The Engine follows the same idea as Anomalib's `Engine`, one level up: the Lightning `Trainer` becomes one of the frameworks underneath it.

```text
                    Engine   (physicalai.engine, in physicalai-train)
        train · export · load · benchmark  +  callbacks, cancellation, packaging
                                  │
        ┌─────────────────────────┼──────────────────────────┐
   physicalai                  lerobot                  third-party
   Lightning, first-party      runs lerobot-train's     (xiaomi, openpi, …)
   policies                    own entry point          separate packages
        └─────────────────────────┼──────────────────────────┘
                                  ▼
        Runtime policy package (manifest.json + native model) ──► InferenceModel
```

Each piece builds on an extension point that already exists:

| Piece     | Contract                                                        | Builds on                                                                                                       |
| --------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Input     | LeRobot-format dataset on disk                                  | Studio dataset snapshots, which training already receives as `dataset_root`                                     |
| Execution | Train, report progress, and stop on request                     | Studio's `TrainingBackend` protocol and the remote trainer, which already runs each job in a spawned subprocess |
| Output    | `exports/<framework>/manifest.json` and the native model folder | Runtime's `policy_package` manifest and Studio's model import                                                   |
| Loading   | A Runtime adapter for the native folder                         | Runtime's `physicalai.inference.adapters` entry-point group                                                     |
| Discovery | An entry point and a settings schema for the GUI                | Robot catalog plugins and the GUI's schema-driven robot forms                                                   |

## Usage

### LeRobot Users

In Python, with LeRobot's own configuration objects:

```python
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig

from physicalai.engine import Engine

cfg = TrainPipelineConfig(
    dataset=DatasetConfig(repo_id="lerobot/pusht"),
    policy=ACTConfig(push_to_hub=False),
    steps=20_000,
)

engine = Engine()
run = engine.train(cfg)  # LeRobot's own train(); the framework is picked from the config type
package = engine.export(run)  # manifest.json plus LeRobot's pretrained_model/
model = engine.load(package)  # Runtime InferenceModel
```

On the command line, with `lerobot-train`'s exact flags passed through unchanged:

```bash
physicalai train lerobot --policy.type=act --dataset.repo_id=lerobot/pusht --steps=20000 --policy.push_to_hub=false
physicalai train lerobot --config_path=outputs/train/act_pusht/checkpoints/last/pretrained_model/train_config.json --resume=true
physicalai train lerobot --help  # prints lerobot-train's own help
```

Models trained with plain `lerobot-train`, or published on the Hugging Face Hub, need no retraining:

```python
engine.export("outputs/train/act_pusht/checkpoints/last/pretrained_model")  # plain lerobot-train output
engine.export("lerobot/act_aloha_sim_transfer_cube_human")  # Hub checkpoint
```

### First-Party Policies

First-party policies keep their Lightning objects and go through the same Engine:

```python
from physicalai.data import LeRobotDataModule
from physicalai.policies import ACT

run = engine.train(ACT(), datamodule=LeRobotDataModule(repo_id="lerobot/pusht"), max_epochs=10)
package = engine.export(run, format="openvino")
```

### Robots

Robots need Runtime, the loader, and the framework, but not `physicalai-train` or Lightning:

```bash
pip install physicalai physicalai-lerobot-plugin  # Runtime plus the LeRobot loader
physicalai run --config runtime.yaml  # InferenceModel(export_dir=".../exports/lerobot")
```

## Keeping LeRobot Exactly LeRobot

- **Same entry point.** `lerobot-train` is `lerobot.scripts.lerobot_train:main`, which calls `register_third_party_plugins()` and then `train()`; `train()` parses `sys.argv` through `lerobot.configs.parser.wrap()`. The CLI path calls that same `main()`, so parsing, validation, third-party `lerobot_policy_*` packages, `--policy.path` fine-tuning, `--config_path`, `--resume`, and all defaults behave identically.
- **Flags pass through unchanged.** The Runtime CLI host parses every subcommand with jsonargparse. A positional framework argument followed by an `nargs=argparse.REMAINDER` argument forwards LeRobot's flags, including `--help`, unchanged (checked with jsonargparse 4.48). No change to the Runtime CLI host is needed.
- **`sys.argv` is scoped.** `TrainPipelineConfig.validate()` reads `sys.argv` itself through `parser.get_path_arg("policy")`, `parser.get_cli_overrides("policy")`, and `parser.parse_arg("config_path")`. For the duration of the call, the framework sets `sys.argv` to exactly what LeRobot should see: the forwarded flags on the CLI path, and only the program name on the Python path. Otherwise arguments from our own CLI or from a test runner leak into LeRobot's config.
- **GUI forms use LeRobot's names.** Pydantic generates JSON schemas directly from `ACTConfig`, `DiffusionConfig`, and `TrainPipelineConfig` (checked with LeRobot 0.5.1). The Studio form shows LeRobot's fields and defaults, such as `policy.chunk_size`, `steps`, and `batch_size`, and submits them as LeRobot flags through the same path as the CLI. It can also show the equivalent command.
- **No lock-in.** The package keeps LeRobot's `pretrained_model/` folder, so `ACTPolicy.from_pretrained()` and LeRobot's own tools keep working on it.

## Contracts

### Engine

```python
class Engine:
    def __init__(self, callbacks: Sequence[EngineCallback] = ()) -> None: ...

    def train(self, config: object | Sequence[str], *, framework: str | None = None, **kwargs: Any) -> Run: ...
    def export(self, source: Run | str | Path, output_dir: str | Path | None = None, *, format: str | None = None) -> Path: ...
    def load(self, package: str | Path, **kwargs: Any) -> InferenceModel: ...
    def benchmark(self, package: str | Path, **kwargs: Any) -> BenchmarkResults: ...  # existing physicalai.benchmark
    def frameworks(self) -> list[FrameworkInfo]: ...
```

- `train()` picks the framework from the config type through `Framework.accepts()`, or from `framework=` when the input is a list of flags or is ambiguous.
- `export()` accepts a `Run`, a native checkpoint folder, or a Hub repository ID. `format=None` means the framework's default, which is the native package for third-party frameworks.
- Callbacks receive `on_train_start()`, `on_step(step, total, metrics)`, `on_checkpoint(path)`, and `on_train_end(run)`; `should_stop()` cancels a run.

### Framework

```python
class Framework(ABC):
    name: ClassVar[str]  # "lerobot"

    @abstractmethod
    def accepts(self, config: object) -> bool: ...  # route native config objects to this framework

    @abstractmethod
    def policies(self) -> list[PolicyInfo]: ...  # catalog for the CLI and GUI

    @abstractmethod
    def config_schema(self, policy: str) -> dict[str, Any]: ...  # GUI form, built from the native config class

    @abstractmethod
    def train(self, config: object | Sequence[str], hooks: TrainHooks) -> Run: ...  # native config or native flags

    @abstractmethod
    def export(self, source: Run | str | Path, output_dir: Path, format: str | None = None) -> Path: ...


class TrainHooks(Protocol):
    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None: ...
    def should_stop(self) -> bool: ...


@dataclass(frozen=True)
class PolicyInfo:
    id: str  # "<framework>/<policy>", for example "lerobot/act"
    display_name: str
    description: str = ""
    license: str = ""
    min_vram_gb: float | None = None
    hub_repos: tuple[str, ...] = ()


@dataclass(frozen=True)
class Run:
    framework: str
    output_dir: Path
    checkpoint: Path  # native model folder, for example checkpoints/last/pretrained_model
```

The rules that keep the contract clean:

- Modules imported during discovery use only the standard library and pydantic; heavy imports happen inside `train()` and `export()`.
- Requests and results are serializable, so the same call works in-process, in a spawned worker, or through the remote trainer's HTTP API.
- `physicalai.engine` cannot live under `physicalai.train` or `physicalai.policies`, because their `__init__` modules import Lightning, torch, and LeRobot.

### LeRobot Framework

```python
class LeRobotFramework(Framework):
    name = "lerobot"

    def accepts(self, config: object) -> bool:
        cls = type(config)
        return cls.__qualname__ == "TrainPipelineConfig" and cls.__module__.startswith("lerobot.")

    def train(self, config: object | Sequence[str], hooks: TrainHooks) -> Run:
        from lerobot.scripts import lerobot_train
        from lerobot.utils.import_utils import register_third_party_plugins

        with lerobot_argv(config), report_steps(lerobot_train, hooks):
            if isinstance(config, Sequence):
                lerobot_train.main()  # lerobot-train's own entry point
            else:
                register_third_party_plugins()
                lerobot_train.train(config)  # the same function, without CLI parsing
        return Run(framework="lerobot", output_dir=..., checkpoint=...)
```

`report_steps()` wraps `lerobot_train.update_policy()`, which `train()` resolves at call time, to report the step and its metrics and to raise when `should_stop()` is true. It is a contained, version-pinned patch until LeRobot offers a callback hook, which we should propose upstream. `sys.argv` and the patch are process-wide, so one LeRobot training runs per process. Studio already runs each job in its own process.

### Package Layout

```text
outputs/train/act_pusht/                    # LeRobot's own output layout
├── checkpoints/last/pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── train_config.json                   # fully resolved TrainPipelineConfig
│   └── processor files
└── exports/lerobot/
    ├── manifest.json                       # "model": {"artifacts": {"lerobot": "pretrained_model"}}
    └── pretrained_model/                   # copy of the checkpoint folder
```

### Runtime Loader

The loader is a regular `RuntimeAdapter` registered in the existing `physicalai.inference.adapters` group. For LeRobot it loads the folder with `PreTrainedPolicy.from_pretrained()` and `make_pre_post_processors()`, and keeps LeRobot's inference behavior (action queue, observation history, and temporal ensembling) by calling LeRobot's own `select_action()` and `reset()`. Returning one action per call is faithful but runs the model at every control step; returning whole chunks needs observation history in the adapter and is a later optimization.

### Plugin Packaging

A third-party plugin is one package. The loader half is the base install for robots, and the training half sits behind an extra:

```toml
[project]
name = "physicalai-acme-policies"
dependencies = ["physicalai>=0.3", "acme-robotics>=1.2"]  # loader side, for robots

[project.optional-dependencies]
train = ["physicalai-train>=0.4"]  # training side

[project.entry-points."physicalai.engine.frameworks"]  # read by physicalai-train
acme = "physicalai_acme_policies.framework:register"

[project.entry-points."physicalai.inference.adapters"]  # read by Runtime (existing group)
acme = "physicalai_acme_policies.inference:register"
```

The plugin's inference module must not import `physicalai-train`, so robots stay lean. `physicalai-train` registers its built-in frameworks (`physicalai` and `lerobot`) through the same group, as it already does for its Torch and ExecuTorch adapters.

## Placement

```text
physicalai (Runtime)        CLI host, InferenceModel, adapter registry, manifest format
   ▲
physicalai-train            physicalai.engine: Engine, Framework, registry, `physicalai train`
   │                        built-in frameworks: physicalai (Lightning) and lerobot
   ▲                  ▲
plugin[train]      Studio backend and GUI (call physicalai.engine only)

plugin (base) ──► physicalai (Runtime) only: the loader, for robots
```

- The LeRobot loader belongs on the deployment side. Runtime's `packages/physicalai-lerobot-plugin` already bridges LeRobot robots and teleoperators into Physical AI and already depends on `lerobot>=0.6.0`, so it is the natural home: a robot installs one package for LeRobot hardware and LeRobot policies.
- This matches the rule in the modular-packages proposal: deployment-artifact loaders belong with inference, and training code belongs in `physicalai-train`.
- Existing APIs stay. `physicalai.train.Trainer` and `physicalai fit|validate|test|predict|export|benchmark` remain the Lightning layer, which the `physicalai` framework uses. The `LeRobotPolicy` Lightning wrapper becomes optional.

## Studio Integration

The backend changes are generic and written once:

- **Policy list.** `GET /api/policies` returns the frameworks' catalogs and settings schemas, and the GUI renders them with the schema form already used for robots (`application/ui/src/features/robots/robot-form/robot-schema/`). This replaces `_POLICY_NAMES`, both `_SUPPORTED_POLICIES` sets, the UI's `MODELS` list, and the PEFT and SnapFlow lists in the backend and UI.
- **Training.** A job stores the framework name and its native settings (for LeRobot, the flags). A training backend runs `Engine.train()` in an isolated worker, reusing the trainer service's spawned-process pattern, and maps Engine callbacks to `ProgressReporter` and `should_stop`. The remote trainer image needs the plugin installed.
- **Model import.** Accept framework export folders and folder-shaped artifacts. Today `_EXPORT_BACKENDS` is hardcoded and the artifact check uses `is_file()`.
- **Live inference.** List inference backends from Runtime's adapter registry instead of the two-value `InferenceBackend` enum.
- **Installation.** The curated `application/backend/src/plugins/manifest.json` gains `policies` entries next to `robots`, so `PluginManager` can install and uninstall them. Policy plugins pull heavy dependencies, so installs should use constraints that keep the environment's torch and transformers builds.
- **IDs.** Policies are identified as `<framework>/<policy>`. Existing models stored with bare names, such as `act`, map to `physicalai/act`.

## Runtime Changes

Two small, generic changes, both useful beyond plugins:

1. `InferenceModel.reset()` also calls a new `RuntimeAdapter.reset()`, which is a no-op by default. Without it, stateful policies keep queues and observation history across episodes. For the POC, a manifest callback that uses `on_load()` and `on_reset()` can stand in.
2. `InferenceModel._get_model_path()` resolves `model.artifacts[backend]` to a path inside the export directory, reusing the containment check in `resolve_artifact()`, so an artifact can be a folder. Today the loader only reads the key and finds files by extension.

`Manifest` already allows extra top-level keys, so provenance (framework name and version, policy ID, and source checkpoint) can be recorded without a schema change. A typed field can follow.

## Trade-Offs

What cannot be identical to LeRobot:

- The command is `physicalai train lerobot ...` instead of `lerobot-train ...`. Flags, defaults, and outputs are the same, and plain `lerobot-train` keeps working.
- `physicalai-train` pins `lerobot==0.6.0` exactly. Users on another LeRobot release hit a resolver conflict, so the LeRobot framework should support a tested version range.
- Robots run policies through Runtime's `InferenceModel`, not LeRobot's control loop. The native folder still works with LeRobot directly.
- One LeRobot training runs per process.
- Intel XPU support depends on LeRobot's own device handling (`lerobot.utils.device_utils` detects XPU), so fixes go upstream instead of into Studio.

What Physical AI gives up compared with the wrapper:

- Studio-only training features (SnapFlow, the PEFT mixins, Studio's image augmentations, and best-`val/loss` checkpoint selection) do not apply to framework-trained models. The framework's own equivalents do, for example LeRobot's `--peft`.
- The GUI training form is defined per framework instead of being one fixed form.
- OpenVINO export still needs a first-party port or framework-side export work, as in the other proposals.
- Frameworks that cannot read LeRobot datasets need a conversion step inside their plugin. Frameworks that cannot share Studio's Python environment (conflicting pins or JAX) need the same contract run in a separate environment, which the design allows but the POC does not build.

## Naming

- Use framework, not backend, for the pluggable layer. Backend already means four things here: export formats (`ExportBackend`), Runtime inference backends (`InferenceModel(backend=...)`), Studio's FastAPI backend, and `TrainingBackend`.
- The module is `physicalai.engine`. `from physicalai import Engine` is not possible, because `physicalai` is a namespace package shared with Runtime.
- The entry-point group is `physicalai.engine.frameworks`.
- The CLI verb is `physicalai train <framework> ...`; `fit` stays the Lightning-level command.

## POC Plan

1. `physicalai.engine`: `Engine`, `Framework`, `TrainHooks`, `Run`, the entry-point registry, and `physicalai train <framework> ...` with `REMAINDER` passthrough.
2. LeRobot framework: training through LeRobot's entry point with scoped `sys.argv` and the step hook, `export()` to a package, a Runtime loader, and `reset()` forwarding through a manifest callback.
3. Tests:
   - The same flags through `lerobot-train` and `physicalai train lerobot` produce the same resolved `train_config.json` (LeRobot writes it into every checkpoint) and the same losses for a few seeded CPU steps.
   - A package loaded with `InferenceModel` returns the same actions as LeRobot's `select_action()`, including across `reset()`.
   - `export()` works for a plain `lerobot-train` output and for a Hub checkpoint.
4. After the POC: the `physicalai` framework (moving `run_training_job()` from `application/backend/src/training/job.py` into the library), then the Studio backend and GUI.

## Open Decisions

- **Names.** `physicalai.engine`, `physicalai.engine.frameworks`, and `physicalai train` versus `physicalai policies train`.
- **LeRobot loader location.** Runtime's `physicalai-lerobot-plugin`, a new package, or `physicalai-train` (which would put `physicalai-train` on robots).
- **GUI paths for LeRobot.** Whether the GUI also offers the Lightning `LeRobotPolicy` path, or only the native LeRobot framework.
- **Trust.** Plugins and `lerobot_policy_*` packages execute code on import. The GUI could label unverified plugins, as robot plugins do with `TRUST_UNVERIFIED_PLUGIN`.

## Relationship to the Other Proposals

| Topic                  | plugin-options.md                      | registry-and-bridge.md                           | This proposal                                                        |
| ---------------------- | -------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| Unit of extension      | A `Policy` subclass                    | A `PolicySpec` plus an `ExternalPolicy` subclass | A `Framework` that trains and exports with its own tooling           |
| Who runs training      | Studio's Lightning `Trainer`           | Studio's Lightning `Trainer`                     | The framework's own trainer; Lightning only for first-party policies |
| Training equivalence   | Proven per named integration           | Proven per framework                             | By construction; only the config mapping and inference need tests    |
| Native user experience | `wrap(native)` keeps the native object | Five hooks around the native model               | Native configs, flags, and artifacts, unchanged                      |
| Discovery              | Entry point to a `Policy` class        | Entry point to `register(registry)`              | Entry point to `register(registry)` for frameworks                   |
| Deployment             | Export through `ExportablePolicyMixin` | `native` package and one generic adapter         | Native package and one Runtime loader per framework                  |
| Separate environments  | Out of scope for v1                    | Inference-only plugins                           | Allowed by the file contract; not in the POC                         |

All three agree on entry-point discovery, one catalog for the backend and UI, and first-party implementations for models that need OpenVINO. The catalog metadata from registry-and-bridge.md (display name, license, minimum VRAM, and Hub repositories) carries over as `PolicyInfo`. This proposal replaces its bridge with the framework contract, and its single generic adapter with one loader per framework.

Checking registry-and-bridge.md against the code also surfaced these feasibility notes, which motivated this proposal:

- The five `ExternalPolicy` hooks miss contracts the trainer relies on: action and observation windows (`reformat_dataset_to_match_policy()` reads `*_delta_indices` from `policy.model`), validation loss (`Policy.validation_step()` calls `compute_val_loss()`, which raises by default, and the GUI always holds out a validation split), checkpoint reloads (Lightning's default `load_from_checkpoint()` cannot restore a model that is built lazily in `setup()`), and an inputs schema (the GUI's `requires_task()` reads the manifest's `input_features`).
- `DatasetFeatures` does not exist yet.
- Porting `LeRobotPolicy` onto the bridge changes its `state_dict` keys from `_lerobot_policy.*` to `model.*`, because its `model` attribute is a property with a no-op setter. Existing checkpoints break unless the keys are remapped.
- A stateless `predict_action_chunk()` does not work for history-based policies: LeRobot Diffusion asserts `n_obs_steps == self.config.n_obs_steps`, and Runtime declares `n_obs_steps` in the manifest but does not stack observation history.
- Runtime's `PolicySource` ignores unknown fields, so Studio cannot read a `policy_id` until Runtime adds the field and Studio updates its pinned Runtime revision.
- The hardcoded lists also include `PEFT_POLICIES` and `SNAPFLOW_POLICIES` in the backend and `peft.ts` and `snapflow.ts` in the UI.
- Registering LeRobot policies by name drops `get_lerobot_policy()`'s fallback to any LeRobot-registered policy unless the registry keeps it.
- A registry module under `physicalai.policies` imports Lightning, torch, OpenVINO, and LeRobot, because `physicalai/policies/__init__.py` imports the LeRobot wrapper eagerly.

## References

Studio:

- `library/src/physicalai/policies/lerobot/policy.py`: `LeRobotPolicy`
- `library/src/physicalai/train/utils.py`: `reformat_dataset_to_match_policy()` and `_get_delta_indices()`
- `library/src/physicalai/data/datamodules.py`: training and validation data loaders
- `library/tests/integration/test_lerobot_wrapper_equivalence.py`: wrapper equivalence tests
- `application/backend/src/training/job.py`: `TrainingJobSpec` and `run_training_job()`
- `application/backend/src/services/training_backends/base.py`: `TrainingBackend` and `TrainingContext`
- `application/backend/src/trainer/`: the remote trainer service and its isolated job runner
- `application/backend/src/services/model_import_service.py`: model import validation
- `application/backend/src/plugins/`: `PluginManager` and the curated plugin manifest

Runtime:

- `src/physicalai/cli/main.py` and `src/physicalai/cli/_spec.py`: the CLI host and `SubcommandSpec`
- `src/physicalai/inference/model.py`: `InferenceModel`
- `src/physicalai/inference/adapters/registry.py`: the adapter registry
- `src/physicalai/inference/manifest.py`: `Manifest` and `PolicySource`
- `packages/physicalai-lerobot-plugin`: LeRobot robots and teleoperators

LeRobot `v0.6.0`:

- `src/lerobot/scripts/lerobot_train.py`: `main()`, `train()`, and `update_policy()`
- `src/lerobot/configs/parser.py` and `src/lerobot/configs/train.py`: CLI parsing and `TrainPipelineConfig.validate()`
- `src/lerobot/datasets/factory.py`: `resolve_delta_timestamps()` and ImageNet statistics
- `src/lerobot/common/train_utils.py`: the checkpoint layout
