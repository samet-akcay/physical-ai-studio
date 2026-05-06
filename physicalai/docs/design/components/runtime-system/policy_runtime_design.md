# Policy Inference and PolicyRuntime Design

This document specifies the concrete design for PhysicalAI's policy inference layer and the per-tick runtime that drives a robot from a policy.

This document translates the gap analysis in `inference_comparison_report.md` into a modular PhysicalAI design. For the broader stack vision this design fits into, see `robot_stack_vision.md`.

---

## 1. Scope

In scope:

- `InferenceModel` API additions
- `InferenceRunner` and the runner family (`SinglePass`, `FlowMatching`, `TemporalEnsemble`)
- `Guidance` and `RTC` (including relative-action reanchoring)
- `Execution` and its concrete implementations (`SyncExecution`, `AsyncExecution`)
- `PolicyRuntime` and the optional runtime components included in the initial implementation
- `ActionQueue` with co-located `ChunkSmoother`
- Shared image-preprocessing `Preprocessor` step composed by per-policy pipelines
- Preflight + runtime config validation
- Timing utilities (`precise_sleep`, overrun warnings)
- First-inference warmup (backend-agnostic)
- Robot thread-safety contract
- CLI subcommand (`physicalai run`)
- Phased rollout and acceptance criteria
- Components that are intentionally deferred until a specific consumer requires them

Out of scope:

- `RobotSystem` design (reserved; see vision doc §7)
- VLM planners, perception stacks, world state, locomotion, whole-body control
- Studio strategy implementations beyond their integration seams
- Manifest schema details beyond illustrative examples
- Detailed RTC math (referenced via the paper)

---

## 2. Design Principles

PhysicalAI is a standalone package. Its primary user is a developer with a robot, a checkpoint, and a Python script. Studio is one downstream consumer; other consumers include CLI users, examples, third-party integrators, evaluation scripts, and CI smoke tests. The runtime API is designed around that standalone use case first.

1. **The initial API should be small and sufficient.** A working policy loop requires `Robot`, `InferenceModel`, `Execution`, `fps`, and optional cameras. Additional constructor arguments are optional and address specific runtime behavior.
2. **Reusable interfaces should have a clear consumer.** Interfaces such as arbiters, filters, and observation assemblers are useful when multiple consumers need the same customization point. Until that need exists, the simpler implementation stays inside `PolicyRuntime` or in the downstream consumer.
3. **Responsibilities should stay within their layer.** `InferenceRunner` decides how the policy computes; `Guidance` decides how denoising is corrected; `Execution` decides when and where the policy runs; `PolicyRuntime` decides how actions reach the robot. Studio strategies compose these pieces for product workflows.
4. **Transport-agnostic execution.** `Execution` covers thread, process, and (future) remote inference under one protocol. The same code path supports today's `application/` `ModelWorker` process pool and a future remote-GPU deployment.
5. **Dict-shaped IO.** `InferenceModel` methods take and return `Mapping[str, Any]`. A documented dictionary contract is sufficient for action chunks and matches existing PhysicalAI style.
6. **Backward compatibility.** Existing users of `ActionChunking`, `select_action`, and the runner family continue to work. The runtime-owned chunking model is added alongside the existing runner-owned model.

---

## 3. Current Baseline

### Inference layer (today)

- `InferenceModel` with `__call__()`, `select_action()`, `reset()`.
- `InferenceRunner` as the internal computation seam.
- `SinglePass` runner.
- `ActionChunking(SinglePass())` runner-level decorator with an internal action deque.
- Manifest-driven runner selection through the component factory.
- Inference-only callbacks: `on_load`, `on_predict_start`, `on_predict_end`, `on_reset`.

### Hardware layer (today)

- `physicalai.robot.Robot` protocol: `connect`, `disconnect`, `get_observation`, `send_action`.
- `physicalai.capture.Camera` ABC: `connect`, `disconnect`, `read`, `read_latest`, `async_read`.
- `physicalai.capture.Frame`: `data`, `timestamp`, `sequence`.
- `physicalai.capture.read_cameras` / `async_read_cameras` for synced multi-camera reads.

### What does not exist today

- `physicalai.runtime` package.
- `PolicyRuntime`.
- `Execution` / `SyncExecution` / `AsyncExecution`.
- `FlowMatching` runner.
- `Guidance` / `RTC`.
- `InferenceModel.predict_action_chunk()`.
- `InferenceModel.close()` and a correct `__exit__` cleanup path.
- `physicalai run` CLI subcommand.

---

## 4. Initial API (Standalone Use)

The initial user-facing API for running a policy on a robot is:

```python
from physicalai.inference import InferenceModel
from physicalai.runtime import PolicyRuntime, SyncExecution
from physicalai.robot.so101 import SO101
from physicalai.capture.cameras.uvc import UVCCamera

model = InferenceModel.load("./exports/act_policy")
robot = SO101(port="/dev/ttyACM0")
front = UVCCamera(index=0)

runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=SyncExecution(mode="chunk"),
    fps=30,
    cameras={"front": front},
)
runtime.run(duration_s=60)
```

This example defines the standalone contract. The runtime can run without an arbiter, filter, assembler, smoother, or callback. Those components are optional and are added to the public API when they are required by a deployment.

The same shape works from the CLI:

```bash
physicalai run --config so101_act.yaml
```

The optional runtime components in §11 are constructor arguments on `PolicyRuntime`. A basic deployment does not need to configure them.

---

## 5. Ownership Rules

```text
InferenceRunner controls how the policy computes actions.
Guidance         controls how denoising is corrected.
Execution        controls when and where the policy runs.
PolicyRuntime    controls how actions are executed on the robot.
Studio strategies control product workflows around the runtime.
```

API additions should preserve these ownership boundaries. If an API needs to cross a boundary, the design should state why that coupling is necessary.

---

## 6. Relationship to LeRobot's Rollout Subsystem

This design closes the gaps identified in PhysicalAI's current inference stack while using PhysicalAI-specific object boundaries.

LeRobot bundles concerns that this design separates. The mapping is one-to-many in both directions:

| LeRobot concept | PhysicalAI equivalent | Why we split it |
|---|---|---|
| `lerobot-rollout` CLI | `physicalai run` subcommand on a new runtime-side `physicalai` console script (§14) | runtime CLI lives in the `physicalai` distribution (no Torch / Lightning); training subcommands plug in via entry points when the training distribution is also installed |
| `RolloutConfig` + `build_rollout_context` | manifest + `PolicyRuntime` constructor + `RuntimeValidator` (§11.6) | DI happens at the runtime constructor; validation is a separate step that runs in tests without a robot |
| `Strategy` (`base` / `sentry` / `highlight` / `dagger`) | `PolicyRuntime` (loop) **+** Studio strategies (workflow composition) | the loop is workflow-agnostic; strategies live in the consumer (Studio), not in this package |
| `InferenceEngine` (`sync` / `rtc`) | `Execution` (`sync` / `async`) **+** `Guidance` (`RTC`) **+** `ActionQueue.smoother` | LeRobot's `RTCInferenceEngine` lumps async dispatch, denoising guidance, queue mechanics, latency tracking, and warmup into one object; this design separates them so RTC works on any flow runner without re-implementing async, and async works without RTC |
| `ActionInterpolator` | deferred (§16) | added when a checkpoint or deployment requires policy-rate ≠ control-rate decoupling |
| `ActionQueue` with delay-aware replace | `ActionQueue` + `RTCQueueMerger` (§11.4) | merge logic is a queue concern; RTC-specific weighting is in `Guidance` |
| `precise_sleep` + overrun warning | `physicalai.runtime.timing` (§11.5) | direct equivalent |
| `compile_warmup_inferences` (Torch-only in LeRobot) | `SupportsWarmup` runner protocol + `Execution.warmup()` (§9) | warmup is backend-agnostic in PhysicalAI; ExecuTorch graph init, OpenVINO `compile_model`, and ONNX session warmup all benefit from running before the control loop starts |
| robot wrapper for thread safety | `Robot` thread-safety contract + optional `ThreadSafeRobotProxy` (§11.7) | the contract is documented first; the wrapper is used for drivers that need serialization |
| relative-action reanchor in RTC | RTC `Guidance` injects `prev_chunk_left_over` reanchored against the snapshot state (§8.1) | reanchoring is RTC-specific; lives with the RTC implementation |
| `VanillaObservationProcessorStep` (single image preprocessor) | shared image-preprocessing `Preprocessor` step in `physicalai/inference/preprocessors/` (Phase 0, §15) | the existing `Preprocessor` namespace gains a shared image step that per-policy preprocessors compose, replacing the duplicated resize/pad/scale logic in `pi05.py` and `smolvla.py` |
| Rerun telemetry | `Callback` example in consumer (§11.3) | telemetry can be implemented through runtime callbacks |
| `PolicyServer` / `RobotClient` (async gRPC) | `RemoteExecution` (§10) + `PolicyServer` (§17) | sibling `Execution` implementation for remote inference deployments; protocol fixed in §10, full design in §17, build target Phase 5 (§15) |
| `Strategy` with arbiter / filter / assembler concerns | simple default behavior in `PolicyRuntime`; product workflows composed by consumers (§19 sketches; deferred runtime-level arbiter / filter protocols in §16) | the standalone API stays focused on the policy loop, while product workflows compose around it |

### Why the boundaries differ from LeRobot

LeRobot's design is appropriate for its package, but it combines concerns that PhysicalAI separates:

1. **`RTCInferenceEngine` mixes math, scheduling, queue, and timing.** A diffusion policy that wants async without RTC, or RTC without a custom engine class, has to subclass or reimplement. In this design, `FlowMatching(guidance=RTC())` + `AsyncExecution()` + `ActionQueue(smoother=...)` compose freely.
2. **`Strategy` bundles loop and workflow.** Adding a new product workflow (e.g. recording with sentry) requires picking one strategy and inheriting from it. In this design, `PolicyRuntime` is the only loop; product workflows live in the consumer.
3. **Engine selection is config + class hierarchy.** In this design, transport (`thread`/`process`/`remote`) is an `Execution` parameter, not a class; the same code path supports the existing `application/` `ModelWorker` (process pool) and a future remote-GPU deployment.

This produces several small PhysicalAI types instead of one large rollout subsystem. The benefit is that each inference-runtime capability can be implemented and tested in isolation.

### Gap-closure traceability

Every gap from the comparison report is closed by an explicit section of this design:

| Gap | Closed by |
|---|---|
| No control-loop runner | §10 `PolicyRuntime` |
| No CLI | §14 `physicalai run` |
| No `InferenceEngine` abstraction | §9 `Execution` |
| No async / RTC | §8 `Guidance` + §9 `AsyncExecution` + §11.4 `RTCQueueMerger` |
| No `precise_sleep` + overrun warning | §11.5 |
| No first-inference warmup | §9 (`Execution.warmup`) |
| No relative-action reanchoring | §8.1 |
| No async gRPC server / client | §10 (`RemoteExecution` protocol position) + §17 (full design; build Phase 5) |
| No Rerun telemetry | §11.3 (`Callback` example in consumer) |
| No shared image-preprocessing step | Phase 0, §15 |
| No preflight assertion | Phase 0, §15 + §11.6 |
| Open-loop chunk consumption (latency spike) | §9 (`Execution.refill_threshold`) + §12 |
| Robot thread-safety wrapper | §11.7 |
| Per-policy image preprocessing drift | Phase 0, §15 |
| Runtime config validation (sync + relative-action, etc.) | §11.6 `RuntimeValidator` |
| Action interpolation (control-rate decoupling) | deferred to §16 until a checkpoint or deployment requires it |
| Strategy abstractions (sentry, dagger, highlight, hil) | handled by downstream consumers rather than by the standalone package |
| Safe-shutdown / return-to-initial | Phase 2 `return_to_home` kwarg on `PolicyRuntime` |

---

## 7. `InferenceModel` API

`InferenceModel` remains the user-facing policy object. The proposed additions are minimal.

```python
class InferenceModel:
    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "InferenceModel": ...

    # Existing
    def __call__(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def select_action(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def reset(self) -> None: ...

    # New
    def predict_action_chunk(
        self,
        observation: Mapping[str, Any],
        *,
        inference_delay: int | None = None,           # in policy frames; for RTC
        prev_chunk_left_over: Mapping[str, Any] | None = None,  # for RTC overlap
    ) -> Mapping[str, Any]: ...
    def close(self) -> None: ...

    def __exit__(self, *args) -> None:
        self.close()
```

### Return shapes

`select_action()` returns a dict with at least:

- `"action"`: `np.ndarray` of shape `(D,)` — the single action.

`predict_action_chunk()` returns a dict with at least:

- `"actions"`: `np.ndarray` of shape `(H, D)` — the chunk.
- `"policy_dt"`: `float | None` — nominal interval between actions, in seconds.
- `"t0"`: `float | None` — observation timestamp the chunk was predicted from.

Additional keys may be present (denoising state, telemetry, RTC overlap state). Consumers (`ActionQueue`, RTC, callbacks) read documented keys; unknown keys are passed through.

The dict shape matches existing PhysicalAI conventions. The design does not introduce a separate action-chunk dataclass.

### Rules

- `inference_delay` and `prev_chunk_left_over` are **runtime-injected** by `Execution` for RTC. End users do not pass them; `Execution` does. Policies that don't support RTC ignore them.
- `close()` deterministically releases model-owned resources (GPU memory, file handles, background threads owned by adapters).
- `InferenceModel` does not expose `start()`, `stop()`, `notify_observation()`, or runtime scheduling methods. Those responsibilities belong to `Execution` and `PolicyRuntime`.

### Compatibility with existing `ActionChunking`

`ActionChunking(SinglePass())` is a runner-level decorator with an internal deque. It continues to work and is **not deprecated**. It and `PolicyRuntime` represent two valid ways to consume chunks:

- **Runner-internal queue (`ActionChunking`):** the runner pops one action per `select_action()` call. Refills synchronously when empty. Suitable for direct `model.select_action(obs)` use without a runtime.
- **Runtime-owned queue (`PolicyRuntime` + `predict_action_chunk()` + `ActionQueue`):** the runtime pops one action per tick. Refills via `Execution`, which controls timing and lets RTC inject overlap state.

Both paths are supported. The runtime path is recommended when running a control loop; the decorator path remains valid for users who call `select_action()` directly.

---

## 8. `InferenceRunner` API

Unchanged in shape. Still the internal seam behind `InferenceModel`.

```python
class InferenceRunner(ABC):
    @abstractmethod
    def run(self, adapter: RuntimeAdapter, inputs: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...
```

Optional structural protocols:

```python
@runtime_checkable
class SupportsWarmup(Protocol):
    def warmup(self, adapter: RuntimeAdapter, sample_inputs: Mapping[str, Any], n: int = 2) -> None: ...

@runtime_checkable
class RequiresAdapter(Protocol):
    def bind(self, adapter: RuntimeAdapter) -> None: ...
```

### Runner family

| Runner | Purpose | Status |
|---|---|---|
| `SinglePass` | one adapter call | exists |
| `ActionChunking` | runner-internal chunk deque (Decorator) | exists; kept permanently |
| `FlowMatching` | iterative flow/diffusion sampling; accepts a `guidance` argument | new |
| `TemporalEnsemble` | ACT-style overlapping chunk smoothing; wraps an inner runner | new |

### `TemporalEnsemble` is not a replacement for `ActionChunking`

These are orthogonal — they solve different problems:

| | Calls model how often? | Combines chunks? | Use when |
|---|---|---|---|
| `ActionChunking` | once per chunk (sparse) | no | inference is expensive; the runner-internal deque is sufficient |
| `TemporalEnsemble` | every tick (dense) | yes (e.g. exponentially-weighted average over overlapping predictions) | inference is cheap; want maximum smoothness |
| `PolicyRuntime` + `predict_action_chunk` + `ActionQueue` | per `Execution.refill_threshold` (sparse, but background-refilled) | optionally (via `ActionQueue.smoother`) | running a control loop |

`TemporalEnsemble` is a **new** runner. It does not subsume `ActionChunking`.

---

## 9. `Guidance` and RTC

Guidance modifies an iterative denoising trajectory. It belongs to the policy computation layer rather than the runtime scheduling layer.

```python
class Guidance(ABC):
    @abstractmethod
    def step(self, denoise_state: DenoiseState) -> DenoiseState: ...

    def reset(self) -> None: ...
```

`RTC` is one concrete `Guidance`:

```python
runner = FlowMatching(guidance=RTC())
```

Rules:

- `RTC` is **only** valid on runners that expose a denoising loop (today: `FlowMatching`).
- `RTC` is **not** an `Execution` mode. Async dispatch and overlap guidance are independent.
- `RTC` with no overlap prefix must be an identity operation. This is a tested invariant.
- ACT-style policies (`SinglePass`, `TemporalEnsemble(SinglePass())`) do not take guidance.
- RTC follows the reverse-time convention end-to-end (`x1_t = x_t - time * v_t`, update is `v_t - guidance_weight * correction`). This is pinned by test.

### 9.1 Relative-action reanchoring

OpenPI-style Pi0 / Pi0.5 policies emit deltas relative to a snapshot of robot state captured at chunk-prediction time. RTC overlap on a relative-action policy must re-express the previous chunk's leftover (computed against snapshot `s_prev`) into the current snapshot's frame `s_curr` before passing it to denoising. Without this, naive RTC drifts on every chunk transition.

Where this lives:

- **`InferenceModel` exposes** an optional `is_relative_action: bool` and an optional `snapshot_state(observation) -> ndarray` method on policies that produce relative actions.
- **`RTC.reset()`** clears any cached snapshot.
- **`Execution`** captures the snapshot at request time and stores it on the in-flight request.
- **`RTCQueueMerger`** (§11.4) reanchors the leftover when the new chunk arrives, before pushing into `ActionQueue`.
- **`RuntimeValidator`** (§11.6) rejects `SyncExecution` for relative-action policies because sync mode has no leftover-and-snapshot bookkeeping.

The bookkeeping lives in `Execution` and the merger; the math lives in the policy and `RTC`.

---

## 10. `Execution` API

`Execution` decides **when** and **where** the runtime asks the model for actions.

```python
class Execution(ABC):
    @abstractmethod
    def start(self, action_queue: "ActionQueue", model: InferenceModel) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def maybe_request(self, observation: Mapping[str, Any]) -> None: ...

    def warmup(self, sample_observation: Mapping[str, Any], n: int = 2) -> None:
        """Run n warmup inferences before the control loop starts. Default no-op.

        First-inference cost is backend-dependent: ExecuTorch graph
        initialization, OpenVINO `compile_model`, ONNX session warmup, and
        kernel/memory allocation are all paid on the first call. Running
        warmup before the loop starts keeps the first real tick from
        absorbing that cost. AsyncExecution implementations run warmup on
        the worker thread or process."""

    def estimated_delay_frames(self, control_dt: float) -> int:
        """Return the running estimate of inference delay in control frames.
        Used by RTC to compute the prefix-mask schedule. Default 0."""
```

### Coupling

`Execution` receives the `ActionQueue` and model at `start()` time. It does not hold a reference to the whole `PolicyRuntime`. This keeps the interface narrow because `Execution` only needs to push completed chunks into the queue and call the model. Tests can drive an `Execution` with a fake queue and a fake model without constructing a runtime.

### Concrete implementations

| Implementation | Where the model runs | When `maybe_request` does work |
|---|---|---|
| `SyncExecution(mode="single_action")` | inline on control thread | every tick |
| `SyncExecution(mode="chunk")` | inline on control thread | when `ActionQueue` needs refill |
| `AsyncExecution(transport="thread")` | worker thread in same process | when queue needs refill and no request is in flight |
| `AsyncExecution(transport="process")` | worker process via queues | same as above; uses `mp.Queue` to hand off observations and chunks |
| `RemoteExecution` | RPC to a remote inference service | same as above; uses a network transport (gRPC, HTTP, etc.) and a separate `PolicyServer` process |

Rules:

- `Execution` is **transport-agnostic**. Threaded async, multi-process async, and remote async are all `Execution` implementations. This matters because consumers (such as `application/`) may run inference in a separate process.
- `AsyncExecution` may own a request queue, but the runtime-owned `ActionQueue` remains the only action queue.
- `AsyncExecution` snapshots observations at request time. In-flight chunks are not cancelled by default.
- Background-worker exceptions surface on the runtime thread on the next `maybe_request` or `pop` call.
- `SyncExecution.mode` is explicit in config. The runtime does not auto-pick between `select_action` and `predict_action_chunk`.
- **Refill threshold is explicit.** `Execution` takes a `refill_threshold` (default: half the chunk horizon). It requests a new chunk when `ActionQueue.depth() < refill_threshold` **and** no request is in flight. This eliminates the periodic latency spike that runner-internal `ActionChunking` produces every `chunk_size` ticks.
- **Latency tracking is `Execution`'s job, not the model's.** `AsyncExecution` keeps a rolling estimate of inference latency and exposes it via `estimated_delay_frames()` so `RTC` can compute its prefix mask correctly. The model never sees wall-clock time.
- **Warmup runs before the loop.** `PolicyRuntime` calls `execution.warmup(sample_obs, n)` after `start()` and before the first tick. `AsyncExecution` runs warmup on the worker thread or process so the first real request does not absorb backend-specific first-inference cost (ExecuTorch graph init, OpenVINO `compile_model`, ONNX session warmup, kernel allocation).

### Process-transport note

`AsyncExecution(transport="process")` is the path that maps cleanly onto a process-pool inference worker (e.g. today's `ModelWorker` + `ModelWorkerRegistry` in `application/`). The `Execution` instance holds references to the worker's `observation_queue` and `output_queue`; `maybe_request` puts an observation snapshot on `observation_queue`; a small reader thread (or a poll on each `maybe_request`) drains `output_queue` and pushes completed action chunks into the `ActionQueue`.

### Remote-transport note

`RemoteExecution` is a sibling of `AsyncExecution`, not a transport flag on it. Its design is detailed in §17. `RemoteExecution` is a sibling rather than a `transport="remote"` parameter because remote inference carries semantics that thread/process transports do not:

- **Endpoint configuration.** Address, authentication, transport (gRPC, HTTP/2, etc.), TLS material, and per-request deadlines are all `RemoteExecution`-specific constructor fields. Folding them into `AsyncExecution` would make the thread/process constructor surface area harder to read and harder to type-check.
- **Reconnection and error semantics.** Network transports surface failure modes — connection loss, partial responses, RPC-deadline exceeded, server restart mid-request — that thread/process transports do not have. `RemoteExecution` owns its own reconnection policy, retry budget, and "what to do with the in-flight chunk on disconnect" behavior. These behaviors do not belong in `AsyncExecution`.
- **Version negotiation.** A remote `PolicyServer` runs an independently-versioned PhysicalAI build. `RemoteExecution.start()` performs a handshake (model identity, action and observation schema, supported guidance) before the first request. Local transports skip this entirely.
- **Server-side component.** A working remote deployment requires a `PolicyServer` process that loads the `InferenceModel` and serves `predict_action_chunk` / `select_action`. That server is part of the runtime package and ships alongside `RemoteExecution`. A transport flag on `AsyncExecution` would still need this server but would hide the requirement.

Build is targeted for Phase 5 (§15) — not Phase 0 or Phase 1. The protocol position is fixed in §10 and the design is fixed in §17 so consumers can plan around it and so the `Execution` interface does not need to grow when the implementation lands.

---

## 11. `PolicyRuntime` API

`PolicyRuntime` lives in `physicalai.runtime` and owns the per-tick loop.

```python
class PolicyRuntime:
    def __init__(
        self,
        robot: Robot,
        model: InferenceModel,
        execution: Execution,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        # Optional runtime components (§11.x):
        action_queue: ActionQueue | None = None,
        callbacks: Sequence[Callback] = (),
        camera_timeout_s: float = 1.0,
        return_to_home: bool = False,
    ): ...

    def run(self, *, duration_s: float | None = None, num_episodes: int | None = None) -> None: ...
    def stop(self) -> None: ...

    @classmethod
    def from_config(cls, path: str | Path) -> "PolicyRuntime": ...
```

`from_config` is the documented programmatic factory for constructing a fully-wired `PolicyRuntime` from a YAML/JSON manifest. It is the same construction path that the `physicalai run` CLI subcommand uses, exposed as a Python entry point so consumers (notebooks, eval scripts, Studio's `application/backend/`, integration tests) can build a runtime from a config file without going through a subprocess.

`from_config` is a thin wrapper over the same jsonargparse parser used by §14: it reads the manifest, instantiates `Robot`, cameras, `InferenceModel`, `Execution`, `ActionQueue`, callbacks, and `PolicyRuntime` via `class_path` / `init_args`, runs `RuntimeValidator.validate(runtime)`, and returns the constructed runtime. The caller controls lifecycle (`runtime.run(...)`, `runtime.stop()`).

### What it owns

- target FPS and per-tick timing
- observation assembly (inline; default shape below)
- the canonical `ActionQueue`
- dispatch to `Robot`
- episode lifecycle
- callback dispatch
- deterministic shutdown (including `return_to_home` if set)

### Responsibilities outside `PolicyRuntime`

- policy math
- denoising
- recording, telemetry, dataset writing (callbacks)
- teleop / arbitration / HIL state machines (consumer concerns; see §16)
- VLM planning, perception, world state, locomotion (future `RobotSystem`)

### Default observation shape

Inline in `PolicyRuntime`:

```python
def _assemble_observation(self) -> dict[str, Any]:
    obs = dict(self.robot.get_observation())
    if self.cameras:
        synced = read_cameras(self.cameras, timeout_s=self.camera_timeout_s)
        obs["images"] = {name: f.data for name, f in synced.frames.items()}
    return obs
```

Consumers that need a different shape, such as dataset-row and telemetry views for Studio, can assemble that view in a `Callback` from `on_observation`. The initial runtime API does not include a pluggable `ObservationAssembler` protocol.

### Per-tick loop (pseudocode)

```python
def run(self, ...):
    with connect(self.robot):
        self.model.reset()
        self.execution.start(self.action_queue, self.model)
        self.execution.warmup(self._sample_observation())
        self.callbacks.on_runtime_start()

        try:
            while not self.should_stop():
                step = StepContext(t=now())
                self.callbacks.on_step_start(step)

                obs = self._assemble_observation()
                self.callbacks.on_observation(obs, step)

                self.execution.maybe_request(obs)
                action = self.action_queue.pop_or_none()

                if action is None:
                    action = self._hold_position()  # last-action or zero-velocity hold

                self.callbacks.before_send_action(action, step)
                self.robot.send_action(action)
                self.callbacks.on_action_sent(action, step)

                self.callbacks.on_step_end(step)
                self.sleep_until_next_tick()
        finally:
            if self.return_to_home:
                self._return_to_home()
            self.callbacks.on_runtime_stop()
            self.execution.stop()
```

### 11.1 `ActionQueue`

`ActionQueue` is a concrete class, not a protocol. It owns the in-flight chunk and emits one action per tick.

```python
class ActionQueue:
    def __init__(self, smoother: ChunkSmoother | None = None): ...
    def push_chunk(self, chunk: Mapping[str, Any]) -> None: ...
    def pop_or_none(self) -> np.ndarray | None: ...
    def depth(self) -> int: ...
```

`push_chunk` accepts the dict returned by `predict_action_chunk()`. The queue reads the documented keys (`actions`, `policy_dt`, `t0`).

### 11.2 `ChunkSmoother` (co-located with `ActionQueue`)

Cross-chunk blending lives in `physicalai.runtime.action_queue` next to `ActionQueue`. The protocol is small:

```python
class ChunkSmoother(Protocol):
    def blend(
        self,
        old_tail: np.ndarray | None,
        new_chunk: np.ndarray,
    ) -> np.ndarray: ...
```

The default behavior is pass-through with no smoothing. The built-in `LerpChunkSmoother(duration_frames=N)` ports today's `QueueMixer` behavior.

Three smoothing concepts stay cleanly separated:

| Concern | Lives in |
|---|---|
| in-denoising overlap correction | `Guidance` (e.g. `RTC`) |
| chunk-vs-chunk ensembling at policy time | `Runner` (e.g. `TemporalEnsemble`) |
| cross-chunk action blending at queue time | `ActionQueue.smoother` (e.g. `LerpChunkSmoother`) |

### 11.3 `Callback`

One callback class spans inference and runtime hooks.

Inference hooks (fired by `InferenceModel`):

- `on_model_load`, `on_predict_start`, `on_predict_end`, `on_model_reset`

Runtime hooks (fired by `PolicyRuntime`):

- `on_runtime_start`, `on_runtime_stop`
- `on_episode_start`, `on_episode_end`
- `on_step_start`, `on_step_end`
- `on_observation`
- `before_send_action`, `on_action_sent`
- `on_user_event`
- `on_error`

The same callback instance can register for both. `before_send_action` and `on_action_sent` are both shipped because recording and telemetry consumers need different timing.

Examples of callback consumers are:

- A `RerunCallback` for live debugging. Telemetry can be implemented as a callback rather than as runtime logic.
- A consumer-side `RecordingCallback` for dataset writing.
- A consumer-side `ReportingCallback` that pushes to an `mp.Queue` for WebSocket reporting.

### 11.4 `RTCQueueMerger`

When RTC is active, the new chunk must be merged with the leftover tail of the previous chunk (delay-aware drop of the inference-delay prefix; reanchoring for relative-action policies). This is a queue-level merge concern, exposed as a kwarg on `ActionQueue`:

```python
queue = ActionQueue(merger=RTCQueueMerger())
```

`RTCQueueMerger` is the only specialized built-in `QueueMerger`. The default merger is `ReplaceMerger`, where the new chunk replaces the old chunk.

### 11.5 Timing utilities

`physicalai.runtime.timing.precise_sleep(target_t)` sleeps until a target monotonic time with sub-millisecond accuracy. `PolicyRuntime` uses it in `sleep_until_next_tick()` and logs an overrun warning when `now() > target_t + tolerance`.

### 11.6 `RuntimeValidator`

Preflight + runtime validation lives in a dedicated module so it can be invoked in tests without a robot:

```python
class RuntimeValidator:
    @staticmethod
    def validate(runtime: PolicyRuntime) -> None: ...
```

Checks include:

- Sync execution is rejected for relative-action policies.
- Refill threshold is feasible given chunk horizon and inference delay (`d ≤ H - s`).
- Camera timeout is less than `1 / fps`.
- Image preprocessor output shape matches model input shape (delegates to §15 preflight).

`RuntimeValidator.validate(runtime)` runs in `PolicyRuntime.__init__` by default; consumers can call it standalone in tests.

### 11.7 `Robot` thread-safety contract

The `Robot` protocol is documented as **not thread-safe by default**. `AsyncExecution` reads observations from a worker thread/process; if the concrete `Robot` is not thread-safe, `ThreadSafeRobotProxy(robot)` wraps it with a per-method lock. The proxy is opt-in:

```python
runtime = PolicyRuntime(
    robot=ThreadSafeRobotProxy(SO101(...)),
    ...
)
```

A robot driver may declare itself thread-safe by exposing `THREAD_SAFE: ClassVar[bool] = True`; `RuntimeValidator` warns when an `AsyncExecution` runs against a robot without this declaration and without the proxy.

---

## 12. Chunking Model

Today, `ActionChunking` owns chunk consumption inside the runner. The new design adds a second path that moves consumption to `PolicyRuntime`. Both paths coexist permanently.

| Concern | Runner-internal path (today, kept) | Runtime path (new) |
|---|---|---|
| chunk production | `model.select_action()` returns queued items | `model.predict_action_chunk()` |
| chunk storage | runner-internal deque (in `ActionChunking`) | `PolicyRuntime.action_queue` |
| chunk consumption | runner pops one per `select_action` call | runtime pops one per tick |
| cross-chunk blending | not supported | `ActionQueue.smoother` |
| request scheduling | implicit; runner refills synchronously when empty | `Execution.maybe_request` (background refill) |
| RTC support | no | yes (via `Guidance` + `RTCQueueMerger`) |

Users pick whichever fits:

- Direct `model.select_action(obs)` use without a control loop → `ActionChunking`.
- Running a control loop with FPS, refill timing, or RTC → `PolicyRuntime` + `predict_action_chunk()`.

The design does not deprecate `ActionChunking` and does not change existing manifest behavior. Manifests using `ActionChunking` continue to work.

---

## 13. Configuration Examples

Configuration uses jsonargparse `class_path` / `init_args` form. `type` shorthand also works (LeRobot-compatible), but the canonical form is shown here.

ACT-style sync:

```yaml
model:
  runner:
    class_path: physicalai.inference.runners.SinglePass

runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    execution:
      class_path: physicalai.runtime.execution.SyncExecution
      init_args:
        mode: chunk
```

ACT-style with temporal ensembling:

```yaml
model:
  runner:
    class_path: physicalai.inference.runners.TemporalEnsemble
    init_args:
      inner:
        class_path: physicalai.inference.runners.SinglePass
      horizon: 100
      ensemble_window: 8

runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    execution:
      class_path: physicalai.runtime.execution.SyncExecution
      init_args:
        mode: single_action
```

Flow policy with RTC, async via process pool:

```yaml
model:
  runner:
    class_path: physicalai.inference.runners.FlowMatching
    init_args:
      guidance:
        class_path: physicalai.inference.guidance.RTC

runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    execution:
      class_path: physicalai.runtime.execution.AsyncExecution
      init_args:
        transport: process
    action_queue:
      class_path: physicalai.runtime.action_queue.ActionQueue
      init_args:
        smoother:
          class_path: physicalai.runtime.action_queue.LerpChunkSmoother
          init_args:
            duration_frames: 30
        merger:
          class_path: physicalai.runtime.action_queue.RTCQueueMerger
```

---

## 14. CLI

`physicalai run` is a subcommand of a new `physicalai` console script that lives in the **runtime distribution** (`physicalai/`). It mirrors `PolicyRuntime.run()` directly.

```bash
physicalai run --config so101_act.yaml
physicalai run --config so101_act.yaml --duration-s 60
physicalai run --config so101_act.yaml --num-episodes 10
```

### Why the CLI lives in the runtime distribution, not in `library/`

The existing `physicalai` console script in `library/pyproject.toml` is a `LightningCLI` subclass at `library/src/physicalai/cli/cli.py`. Its module-level imports (`lightning.pytorch.cli`, `physicalai.train.Trainer`, `physicalai.policies.base.Policy` which extends `LightningModule`) pull `lightning` and `torch` at import time. **`physicalai run` cannot live there** because the runtime is shipped to inference / edge hosts that do not have Torch or Lightning installed and cannot install them: Torch is not edge-friendly to deploy, and pulling Lightning for a runtime-only host wastes hundreds of MB and a binary dependency we explicitly avoid.

The runtime distribution (`physicalai/pyproject.toml`) lists no Torch and no Lightning today (deps are `numpy`, `loguru`, `pydantic`, `pyyaml`, `opencv-python-headless`, `transformers`, `safetensors`, `onnxruntime`, `openvino`, `openvino_tokenizers`, plus capture/robot extras). The runtime CLI must preserve this constraint.

### Layering: one `physicalai` command, two distributions, plugin discovery

The runtime distribution owns the entry point. Training subcommands (the existing `fit`, `validate`, `test`, `predict`, `benchmark`) plug in via `[project.entry-points]` discovery when the training distribution is also installed.

**Runtime distribution (`physicalai/pyproject.toml`):**

```toml
[project.scripts]
physicalai = "physicalai.cli.main:main"

[project.entry-points."physicalai.cli.subcommands"]
run     = "physicalai.cli.run:register"
serve   = "physicalai.cli.serve:register"  # Phase 5; see policy_server_design.md
```

**Training distribution (`library/pyproject.toml`):** drops `physicalai` from its `[project.scripts]` and registers training subcommands as plugins:

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit       = "physicalai.train.cli:register_fit"
validate  = "physicalai.train.cli:register_validate"
test      = "physicalai.train.cli:register_test"
predict   = "physicalai.train.cli:register_predict"
benchmark = "physicalai.benchmark.cli:register"
```

The training-side `register_*` functions are the only place `LightningCLI` and Torch are imported. They are not loaded by the runtime CLI when only the runtime distribution is installed because `importlib.metadata.entry_points()` returns an empty group for `physicalai.cli.subcommands` from training.

### CLI implementation

`physicalai.cli.main:main` is a thin dispatcher built on plain **`jsonargparse`** (no Lightning):

1. Discover subcommands via `importlib.metadata.entry_points(group="physicalai.cli.subcommands")`.
2. For each, call its `register(subparsers)` to add a parser.
3. Parse argv, dispatch to the chosen subcommand's handler.

`jsonargparse` is added as a runtime dependency in `physicalai/pyproject.toml`. It is pure Python with no Torch / Lightning footprint and provides the same `class_path` / `init_args` config style the design already uses for manifests.

### `physicalai run` subcommand

`physicalai.cli.run:register` defines a `jsonargparse.ArgumentParser` that takes a YAML config and instantiates `PolicyRuntime` via `class_path` / `init_args`. Its handler:

1. Loads the manifest.
2. Constructs `Robot`, cameras, `InferenceModel`, `Execution`, `PolicyRuntime` via `class_path` / `init_args`.
3. Runs `RuntimeValidator.validate(runtime)`.
4. Calls `runtime.run(duration_s=..., num_episodes=...)`.
5. Handles `SIGINT` / `SIGTERM` by calling `runtime.stop()` and waiting for clean shutdown.

The subcommand owns no per-tick logic. It is a thin DI + lifecycle wrapper over `PolicyRuntime`. The Python equivalent for in-process consumers is `PolicyRuntime.from_config(path)` (§11): both share the same `jsonargparse` parser and produce an equivalent, validated runtime instance.

### What this means for dev boxes vs edge hosts

| Environment | What's installed | Available subcommands |
|---|---|---|
| Edge / inference host | `physicalai` only | `physicalai run`, `physicalai serve` (Phase 5) |
| Dev box | `physicalai` + `physicalai-train` | `run`, `serve`, `fit`, `validate`, `test`, `predict`, `benchmark` |

One command name. No Torch on edge. Training subcommands appear automatically when the training distribution is present. No code in the runtime distribution imports anything from the training distribution.

---

## 15. Phases and Acceptance Criteria

### Phase 0 — Correctness fixes (independent of the rest)

- LeRobot wrapper hard-fails (or loudly warns) when checkpoint-pinned processor state is missing.
- A shared image-preprocessing `Preprocessor` step (`physicalai/inference/preprocessors/image.py`) with explicit `scale`, `channel_order`, `pad_mode` fields. This step is composed by per-policy preprocessor pipelines; it is not a new top-level abstraction.
- Image preprocessing parity across supported policy families: `pi05.py` and `smolvla.py` refactored so their preprocessor pipelines compose the shared image step instead of carrying duplicated resize/pad/scale logic.
- Preflight log of preprocessing assumptions on first inference; mismatches raise with provenance.

**Accept when**: a known-divergent checkpoint either loads with correct processor state or fails loudly; supported policy families pass parity tests against the reference implementation.

### Phase 1 — Inference API additions

- `InferenceModel.close()` and `__exit__` cleanup.
- `InferenceModel.predict_action_chunk()` with documented dict shape (§7).
- `Guidance` protocol.
- `FlowMatching` runner with denoising seam.
- `TemporalEnsemble` runner.

**Accept when**: a flow policy can be instantiated with `FlowMatching(guidance=RTC())`, `predict_action_chunk()` returns a dict with `actions`, `policy_dt`, `t0`, and `RTC` with no overlap is identity (tested).

### Phase 2 — Runtime core (initial standalone API)

- `physicalai.runtime` package.
- `PolicyRuntime` with the initial constructor (§11): `robot`, `model`, `execution`, `fps`, optional `cameras`, `action_queue`, `callbacks`, `return_to_home`.
- `ActionQueue` + `ChunkSmoother` protocol + `LerpChunkSmoother` (co-located).
- `Execution`, `SyncExecution`, `AsyncExecution(transport="thread")`.
- `StepContext`, `UserEvent`, `Callback` base.
- `RuntimeValidator`.
- `precise_sleep` + overrun warning.
- `Robot` thread-safety contract documentation + `ThreadSafeRobotProxy`.
- `physicalai run` CLI subcommand.

**Accept when**: the §4 standalone example runs end-to-end against a mock robot at the requested FPS within tolerance; shutdown is deterministic; one callback observes both inference and runtime hooks; the CLI subcommand runs the same example from a YAML config.

### Phase 3 — Async transport + RTC

- `AsyncExecution(transport="process")` with single-flight semantics.
- `Execution.warmup()` and `estimated_delay_frames()` wired on async paths.
- `RTC` `Guidance` end-to-end on a flow policy.
- `RTCQueueMerger` with delay-aware drop and relative-action reanchoring.
- `RuntimeValidator` rejects `SyncExecution` for relative-action policies.

**Accept when**: a flow policy runs under `AsyncExecution(transport="process") + RTC()` without chunk-boundary discontinuities; relative-action checkpoints run without drift; warmup amortizes backend-specific first-inference cost so the first control tick stays within the FPS budget.

### Phase 4 — Hardening

- `LerpChunkSmoother` numerical equivalence test against a reference recorded session.
- Latency-tracking accuracy test for `AsyncExecution.estimated_delay_frames()`.
- Background-worker exception surfacing tested.
- Documentation: standalone quickstart, callback authoring guide, `Execution` choice guide.

**Accept when**: smoothing matches the reference within tight tolerance; latency estimate stays within 1 frame of measured ground truth; an unhandled worker exception surfaces on the next runtime call.

### Phase 5 — `RemoteExecution` and `PolicyServer`

See §17 for the full design.

- `RemoteExecution` client implementing the `Execution` contract over gRPC bidirectional streaming.
- `PolicyServer` process serving `predict_action_chunk` / `select_action`, with version handshake, warmup RPC, and structured per-request logs.
- `physicalai serve` CLI subcommand parallel to `physicalai run`, registered as a runtime-distribution entry point under `physicalai.cli.subcommands` (§14).
- RTC compatibility verified end-to-end: `prev_chunk_left_over` and `inference_delay` cross the wire and the server-side `Guidance` consumes them identically to the local async path.
- Failure-mode tests for the table in `policy_server_design.md` §7 (connection loss before/mid stream, deadline exceeded, schema mismatch at handshake, model swap mid-stream).

**Accept when**: a flow policy running under `RemoteExecution + PolicyServer` produces the same actions (within numerical tolerance) as the same policy running under `AsyncExecution(transport="process")`; reconnection within `reconnect_budget_s` recovers without runtime failure; the runtime exits cleanly when the budget is exhausted.

### Phase 6 — Deferred components

See §16. These components are implemented when a concrete consumer needs the behavior and the behavior cannot be expressed through callbacks, custom `Execution` / `Robot` subclasses, or consumer-side composition.

---

## 16. Deferred Components

The components in this section are intentionally not part of the initial runtime API. Each one is a customization point that may be useful in a particular deployment, but each one also commits the package to a public interface that has to be supported long term. Adding any of them increases the surface area of `physicalai.runtime`, increases the number of valid configurations the runtime has to validate, and increases the number of LeRobot-style "look-alike" abstractions that downstream readers have to learn.

The default position is therefore: **do not add these components until a concrete consumer needs the behavior, and the behavior cannot be expressed through callbacks, custom `Execution` / `Robot` subclasses, or consumer-side composition.**

Each subsection documents:

- **What it does** — what the component would do if it existed.
- **Why it is deferred** — why the initial API does not include it.
- **Condition for adding it** — the concrete deployment or consumer requirement that justifies promoting it from this section into the public API.

When a component graduates out of this section, the change should land with a concrete consumer that uses it, a configuration example, and an acceptance test. This keeps each addition tied to deployed behavior rather than to anticipated workflows.

### 16.1 `ObservationAssembler` protocol

**What it does.** A pluggable replacement for the inline `_assemble_observation()` method in `PolicyRuntime` (§11). It would let a consumer swap the entire observation-building strategy — for example, to produce three views per tick (model input, dataset row, telemetry payload) from a single underlying read of the robot and cameras.

**Why it is deferred.** The initial design (§11) assembles a single dict observation inline and exposes it through `Callback.on_observation`. Consumers that need additional views (e.g. a dataset row or telemetry payload) can derive them in a callback. Promoting assembly to a protocol commits the package to a public type that every runtime consumer must understand, when in practice all known consumers can be served by callback-based derivation.

**Condition for adding it.** Two or more independent runtime consumers need a non-default observation shape, the callback-derived views cause duplicated or inconsistent observation-formatting logic across consumers, and the duplicated logic is non-trivial enough that sharing it through a `Callback` mixin is not a clean fit.

### 16.2 `ActionArbiter` protocol

**What it does.** A runtime-owned object that selects, on each tick, which of several candidate actions to send to the robot. Inputs would be the policy action (from the `ActionQueue`) and zero or more alternative sources such as a teleop stream, a hold-position request, or a safety override. Output is a single action passed to `robot.send_action`. LeRobot does not have an explicit `ActionArbiter` class; in LeRobot, arbitration is hardcoded inside each `Strategy` subclass. The proposed protocol would lift that arbitration into a runtime-level abstraction.

**Why it is deferred.** The initial design covers the common case — policy-only — by reading from the `ActionQueue` directly. The two-source case (policy vs. teleop substitution) is handled in the consumer: Studio's `RobotControlWorker` decides per-tick whether to substitute a teleop action before calling the runtime, based on `follower_source` (§18). Promoting arbitration to a runtime-level protocol means every consumer that does not need it still has to construct or accept a default arbiter, and every callback that observes `before_send_action` has to reason about which source produced the action.

**Condition for adding it.** A consumer needs three or more action sources (e.g. policy, teleop, hold-position, safety override) with shared priority logic, **or** two or more runtime consumers reimplement the same two-source arbitration logic and the duplication has caused a divergence bug.

### 16.3 `ActionFilter` / `SafetyGate` protocol

**What it does.** A runtime-owned chain of action transformations that runs between the arbiter (or the `ActionQueue`, if no arbiter) and `robot.send_action`. Examples: joint-limit clamping, velocity caps, workspace boundaries, deadman-switch enforcement, software E-stop. LeRobot does not have an explicit `ActionFilter` class either; today, filtering of this kind lives inside individual robot drivers and processor steps. The proposed protocol would lift these checks into runtime-level composable filters.

**Why it is deferred.** Action filtering today belongs to the `Robot` driver: a driver that owns the hardware also owns the limits the hardware can tolerate. Lifting filters into the runtime risks duplicating limits that drivers already enforce, and it asks the runtime to reason about hardware specifics it does not own. Per-deployment safety rules (workspace boundaries for a particular cell, deadman-switch wiring) are best expressed at the consumer level, where the deployment context is known.

**Condition for adding it.** A safety-critical check has to run identically across multiple `Robot` drivers and multiple consumers, **and** it cannot be expressed as a `Robot` wrapper or a `Callback` that vetoes a tick. The first concrete trigger is likely a multi-robot safety-certification requirement that mandates a single audited code path for limit enforcement.

### 16.4 `ActionInterpolator`

**What it does.** Decouples the policy's natural action rate from the runtime's control rate. For example, a policy that emits actions at 10 Hz running on a robot ticking at 50 Hz: the interpolator would produce four intermediate actions per policy action by linear (or higher-order) interpolation. Distinct from `ChunkSmoother`, which blends across chunk boundaries; the interpolator works inside a chunk to upsample.

**Why it is deferred.** All currently-supported policy / robot combinations operate at matched rates (or are configured to do so). The runtime's `fps` field is a single number for both policy and control; introducing a separate `policy_fps` and an interpolator commits the runtime to two-rate semantics that every runner, every smoother, and every callback has to handle correctly.

**Condition for adding it.** A concrete checkpoint has a fixed policy rate that is meaningfully lower than the robot's required control rate, **and** running the policy at the higher rate is not feasible (cost, latency, or hardware constraint).

### 16.5 `ShutdownPolicy` protocol

**What it does.** Replaces the `return_to_home: bool` constructor kwarg on `PolicyRuntime` with a pluggable shutdown sequence. A `ShutdownPolicy` would be invoked from the `finally` block of `PolicyRuntime.run()` and could implement multi-stage shutdown: e.g. ramp velocity to zero, retract end-effector, move to a configured pose, power down a gripper, log final state.

**Why it is deferred.** A boolean kwarg covers the two cases the initial design needs: leave the robot where it is, or move it to a home pose. Promoting shutdown to a protocol commits the package to a public type and to specifying the contract under failure (what happens if the shutdown sequence itself raises, what happens on `SIGKILL`, what timeout the runtime enforces). The boolean defers all of those questions until a deployment actually needs them.

**Condition for adding it.** A deployment requires a shutdown sequence that is not expressible as `return_to_home: bool` — typically multi-stage (release gripper, then retract, then move to home) or environment-aware (different home pose depending on what the robot was holding) — **and** the sequence has to be parameterized by configuration rather than hardcoded into a `Robot` driver's `disconnect()`.

---

## 17. `RemoteExecution` and `PolicyServer`

`RemoteExecution` is the third concrete `Execution` implementation alongside `SyncExecution` and `AsyncExecution` (§10). It runs the policy on a host separate from the robot, paired with a `PolicyServer` process on the inference host. From the runtime's perspective, `RemoteExecution` is just another `Execution`: it satisfies the same contract, pushes completed chunks into the `ActionQueue`, reports `estimated_delay_frames()` for RTC, and is started and stopped by `PolicyRuntime` exactly like its siblings. `ActionQueue`, `ChunkSmoother`, callbacks, and `Guidance` choice are unchanged.

The full design — process model, server-side responsibilities, gRPC transport, wire schema, RTC compatibility over the wire, failure-mode policy, configuration surface, differences from LeRobot, and out-of-scope items — lives in **[`policy_server_design.md`](./policy_server_design.md)**. This split exists because the server-side material (transport, proto, server lifecycle, auth) is large enough to warrant its own document and is read by a different audience than the runtime contract itself. The two documents are kept in sync: the `Execution` contract `RemoteExecution` satisfies is defined here in §10, and Phase 5 acceptance criteria for the build live here in §15.

The build target is **Phase 5** (§15), not Phase 0 or Phase 1. The protocol position (sibling `Execution` rather than `transport="remote"` flag on `AsyncExecution`) is fixed now, and the design is fixed in `policy_server_design.md`, so the `Execution` interface does not need to grow when the implementation lands and downstream consumers can plan around it.

---

## 18. Mapping to a Downstream Consumer (`application/`)

This section illustrates how one downstream consumer — Studio's `application/backend/` — composes the initial API. It is included as a worked example. The standalone package design is defined by §4–§11.

### Class-by-class mapping

| Today (`application/backend/src/`) | Composition over `physicalai` | Notes |
|---|---|---|
| `workers/robot_control_worker.py:RobotControlWorker` | thin `BaseThreadWorker` shell wrapping a `PolicyRuntime` | keeps WebSocket transport, event translation, and `mp.Queue` reporting; delegates the loop to `PolicyRuntime` |
| `workers/robot_control_worker.py:WorkerEvents` | unchanged | still the API-thread → loop control surface |
| `workers/robot_control_worker.py:RobotControlState` | unchanged | session state for the WebSocket layer |
| `control/environment_integration.py:EnvironmentIntegration` | factory/lifecycle parts (robot client construction, camera startup) stay in `application/`; per-tick observation formatting becomes a consumer-side `ObservationCallback` if needed | `PolicyRuntime` provides a default observation shape; multi-shape Studio output (model + dataset + telemetry) lives in callbacks, not in `physicalai` |
| `control/sync_mixed_model_integration.py:SyncMixedModelIntegration` | replaced by `AsyncExecution(transport="process")` + `ActionQueue(smoother=LerpChunkSmoother(...))` | `InferencePoller` becomes the `Execution` implementation; `QueueMixer` becomes the `ActionQueue.smoother` |
| `control/inference_poller.py:InferencePoller` | folded into `AsyncExecution(transport="process")` | single-flight semantics preserved |
| `control/queue_mixer.py:QueueMixer` | becomes `LerpChunkSmoother` in `physicalai.runtime.action_queue` | LERP duration becomes a config knob |
| `control/inference_result.py:InferenceResult` | replaced by the `predict_action_chunk()` dict shape (§7) | `time` field becomes `t0`; `data` becomes `actions` |
| `workers/model_worker.py:ModelWorker` | unchanged shape; `select_action` call replaced by `predict_action_chunk` | the worker stays a separate process; what changes is the call it makes |
| `workers/model_worker_registry.py:ModelWorkerRegistry` | unchanged | the pre-spawned pool is orthogonal to the runtime design |
| `RecordingMutation` integration | becomes a `RecordingCallback` on `PolicyRuntime`, plus episode-lifecycle hooks | per-frame recording uses `on_action_sent`; episode boundaries use `on_episode_start`/`on_episode_end` driven by user events |
| teleop arbitration via `follower_source` | lives in `RobotControlWorker`'s loop wrapper around `PolicyRuntime` | the `follower_source` switch chooses whether to use the policy's action or substitute teleop before calling `robot.send_action`; a shared `ActionArbiter` can be added later if multiple consumers need the same behavior |
| WebSocket reporting via `mp.Queue` | becomes a `ReportingCallback` that pushes to the queue | runtime emits structured events; the callback formats and forwards |

### Migration shape

After the design lands, the WebSocket entry point in `application/` looks roughly like:

```python
runtime = PolicyRuntime(
    robot=ThreadSafeRobotProxy(env.follower),
    cameras=env.cameras,
    model=model,
    execution=AsyncExecution(
        transport="process",
        worker=model_worker_registry.acquire(model_spec, backend),
    ),
    fps=30,
    callbacks=[
        ReportingCallback(queue),
        RecordingCallback(dataset, recording_mutation),
    ],
)

control_worker = RobotControlWorker(
    runtime=runtime,
    events=events,
    queue=queue,
    follower_source=session.follower_source,  # consumer-side teleop substitution
)
control_worker.start()
```

The `RobotControlWorker` shrinks: it owns the thread, the events, the WebSocket queue, the per-session lifecycle, and any consumer-specific action substitution. Everything per-tick lives in `PolicyRuntime`.

### Migration sequencing (consumer-side; not part of this design's acceptance criteria)

1. Land `PolicyRuntime` with `SyncExecution` only. Use it in the new CLI or test harness, not in `RobotControlWorker` yet.
2. Add `AsyncExecution(transport="process")` with semantics matching `InferencePoller`.
3. Add `LerpChunkSmoother` matching `QueueMixer` behavior. Verify numerical equivalence on recorded sessions.
4. Cut `RobotControlWorker` over to wrapping a `PolicyRuntime`. Keep the WebSocket protocol unchanged.
5. Remove `SyncMixedModelIntegration`, `InferencePoller`, `QueueMixer`, and `InferenceResult` once nothing imports them.

The migration is staged so that at every step `application/` continues to work end-to-end. This sequencing belongs in a separate `application/` migration document once this design is reviewed and stable.

---

## 19. Studio Strategy Composition Sketches

LeRobot ships `Strategy` subclasses (sentry, dagger, highlight, HIL) that bundle the per-tick loop with workflow-level behavior. PhysicalAI's design splits the loop (`PolicyRuntime`, `physicalai.runtime`) from the workflow composition (consumer side). This section shows, for each LeRobot strategy, how the equivalent workflow composes over the initial PhysicalAI runtime API. The sketches are illustrative — they are not part of the `physicalai` package — but they exist so the design's load-bearing claim that "strategies live in the consumer" is verifiable rather than asserted.

The common shape: each strategy is a small object owned by Studio's `application/backend/` (or by another consumer). It holds a `PolicyRuntime`, registers one or more `Callback` instances, and may wrap `runtime.run()` with workflow-level state. None of these objects need additions to `physicalai.runtime` to work.

### 19.1 Sentry

**Goal.** Run the policy continuously until a trigger fires (a detected object, a scene change, an external event), then take a workflow action (start recording, alert an operator, switch to a different policy).

**Composition.**

```python
class SentryStrategy:
    def __init__(self, runtime: PolicyRuntime, trigger: SentryTrigger, on_fire: Callable):
        self.runtime = runtime
        self.trigger = trigger
        self.on_fire = on_fire
        runtime.callbacks.append(SentryCallback(trigger, self._handle_fire))

    def _handle_fire(self, step: StepContext, evidence: Mapping[str, Any]) -> None:
        self.on_fire(step, evidence)  # consumer-defined; e.g. start recording

    def run(self, **kwargs) -> None:
        self.runtime.run(**kwargs)
```

`SentryCallback` evaluates `trigger` in `on_observation` (it sees the observation dict before the runtime hands it to the policy) and calls `_handle_fire` when the trigger matches. The trigger itself is consumer code: a small classifier, a heuristic on observation keys, or an external signal handler.

**What `physicalai` provides.** `Callback.on_observation`, `StepContext`. Nothing else.

**What stays in the consumer.** `SentryStrategy`, `SentryTrigger`, `SentryCallback`, the actual workflow action (start recording, alert, etc.).

### 19.2 HIL (human-in-the-loop)

**Goal.** Interleave policy actions with human teleop. The human can pause the policy, take over, demonstrate a recovery, then return control. A common variant: blend policy and human commands during transitions instead of switching abruptly.

**Composition.**

```python
class HILStrategy:
    def __init__(self, runtime: PolicyRuntime, teleop: TeleopSource):
        self.runtime = runtime
        self.teleop = teleop
        runtime.callbacks.append(HILCallback(teleop, self._mode))

    def _mode(self) -> Literal["policy", "teleop", "blend"]:
        return self.teleop.current_mode()  # consumer-side state machine
```

`HILCallback` implements `before_send_action`. Depending on `_mode()` it either:

- passes the policy's action through unchanged (`"policy"`),
- replaces it with the teleop sample (`"teleop"`),
- or returns a weighted average of the two (`"blend"`).

The callback returns the substituted action; `PolicyRuntime` sends whatever the last `before_send_action` callback returns. (The contract that `before_send_action` may rewrite the action is part of §11.3 and is the same hook §18 / Studio's `RobotControlWorker` uses for `follower_source` substitution.)

**What `physicalai` provides.** `Callback.before_send_action` with action-rewrite semantics. The teleop hardware integration is already in `physicalai.robot` (it is the same `Robot`-or-similar interface the leader arm exposes today).

**What stays in the consumer.** `HILStrategy`, `HILCallback`, the mode state machine, the UI for switching modes, and the recording bookkeeping that distinguishes policy frames from teleop frames in the dataset.

**Note on arbitration.** The HIL case is exactly the "two-source arbitration" case discussed in §16.2. Today it is handled in the consumer's callback. If a third source appears (e.g. a safety override), promoting `ActionArbiter` to a runtime-level protocol becomes warranted — see §16.2's promotion condition.

### 19.3 Highlight

**Goal.** Record continuously, but only persist short clips around interesting events (a successful grasp, a failure, an operator-flagged moment). The runtime keeps a rolling window of recent frames; on a highlight trigger, the surrounding window is written to permanent storage.

**Composition.**

```python
class HighlightStrategy:
    def __init__(
        self,
        runtime: PolicyRuntime,
        window_s: float,
        trigger: HighlightTrigger,
        sink: HighlightSink,
    ):
        self.runtime = runtime
        runtime.callbacks.append(
            RollingBufferCallback(window_s)  # writes to in-memory ring buffer
        )
        runtime.callbacks.append(
            HighlightCallback(trigger, sink)  # on trigger: snapshot the buffer to sink
        )
```

`RollingBufferCallback` records observations and actions to a fixed-size ring buffer (`on_observation`, `on_action_sent`). `HighlightCallback` evaluates the trigger every tick and, when it fires, asks the rolling buffer for its current window and writes it to the sink (disk, S3, dataset shard).

**What `physicalai` provides.** Callback hooks for observation, action, and step boundaries. A rolling buffer and a highlight trigger are not in `physicalai`.

**What stays in the consumer.** Both callbacks, the trigger, the sink, and the policy for what counts as "interesting".

### 19.4 DAgger

**Goal.** Data aggregation for imitation learning: run the policy while a human expert provides simultaneous demonstrations; record both the policy's intended action and the expert's correction; use the aggregated dataset to fine-tune the policy.

**Composition.**

```python
class DAggerStrategy:
    def __init__(
        self,
        runtime: PolicyRuntime,
        expert: TeleopSource,
        dataset: DatasetWriter,
        beta_schedule: BetaSchedule,
    ):
        self.runtime = runtime
        runtime.callbacks.append(
            DAggerCallback(expert, dataset, beta_schedule)
        )
```

`DAggerCallback` implements `before_send_action`:

- It reads the policy's proposed action (the argument to the callback).
- It reads the expert's action from `expert`.
- It records both into `dataset` along with the observation (already available from `on_observation` earlier in the same tick — the callback can hold a reference between hooks via `StepContext`).
- It returns either the policy action, the expert action, or a beta-mixed action depending on `beta_schedule.current()`.

The training side is unchanged: the dataset that `DatasetWriter` produces is consumed by the standard `physicalai-train` flow.

**What `physicalai` provides.** The same callback hooks as the other strategies, plus `StepContext` as the per-tick scratch space for sharing the policy action and the expert action between `on_observation` and `before_send_action` within a single tick.

**What stays in the consumer.** `DAggerCallback`, the beta schedule, the expert input wiring, and the dataset format choice.

### 19.5 What this section demonstrates

For each strategy, the per-tick loop in `physicalai.runtime` is unchanged. The composition uses three already-existing pieces of the public API:

- `PolicyRuntime.callbacks` — a sequence of `Callback` objects the runtime invokes at documented hook points.
- `Callback.on_observation`, `Callback.before_send_action`, `Callback.on_action_sent`, `Callback.on_step_start`, `Callback.on_step_end` — the hooks that strategies attach to.
- `StepContext` — a per-tick scratch object that callbacks can use to share state within a single tick (used by DAgger to share the policy action between two hooks; used by Highlight to correlate observations and actions in the rolling buffer).

The pattern scales because each strategy's complexity lives entirely in its consumer-side callbacks and its consumer-side state machine, not in the runtime. Adding a fifth strategy (e.g. `EvalStrategy` that runs N episodes against a scoring callback) would follow the same shape and would not require changes to `physicalai.runtime`.

If two or more strategies in two or more independent consumers converge on substantially the same composition, that is the trigger for promoting the shared parts into `physicalai` (see §16's general graduation condition). Until then, they live where they are used.

---

## 20. Open Questions

1. **`TemporalEnsemble` output shape.** The current recommendation is one smoothed action per tick rather than a smoothed chunk. This should be revisited if a policy or runtime needs chunk-shaped ensemble output.
2. **`AsyncExecution` cancellation policy.** Today's `InferencePoller` does not cancel in-flight requests. Keep that default; add `cancel_inflight=True` opt-in if a use case appears.
3. **`Execution` reader thread for process transport.** Whether `AsyncExecution(transport="process")` runs a small reader thread to drain the worker's `output_queue`, or polls inside `maybe_request`. Recommendation: small reader thread, because polling-only delays chunk arrival by up to one control tick.
4. **`ObservationAssembler` protocol.** Studio needs model, dataset, and telemetry views of each observation. The current recommendation is to serve that need through callbacks in Phase 2 and add a protocol later only if callbacks cause duplicated or inconsistent observation-formatting logic.
5. **`RemoteExecution` and `PolicyServer` open questions.** Transport choice (gRPC vs HTTP/2+JSON), multi-tenant single-server semantics, and authentication posture are tracked in `policy_server_design.md` §10.2. They are listed there to keep server-side discussion in one place.

---

## 21. File Layout

```text
physicalai/inference/
  runners/
    base.py
    protocols.py
    single_pass.py
    action_chunking.py       # kept; not deprecated
    flow_matching.py
    temporal_ensemble.py
  guidance/
    base.py
    rtc.py
  callbacks/
    base.py
  model.py
  component_factory.py

physicalai/runtime/
  __init__.py
  policy_runtime.py
  action_queue.py            # ActionQueue + ChunkSmoother + LerpChunkSmoother + RTCQueueMerger (co-located)
  execution/
    base.py
    sync.py
    async_thread.py
    async_process.py
    remote/
      __init__.py            # RemoteExecution (client-side Execution implementation)
      client.py              # gRPC client, reader thread, latency tracking
      server.py              # PolicyServer process (loads InferenceModel; serves Predict)
      proto/                 # generated gRPC stubs from policy_server.proto
      auth.py                # AuthConfig, MutualTLSAuth, BearerTokenAuth
      codecs.py              # ImageCodec (JPEG/PNG/AV1) + Tensor (de)serialization
  context.py                 # StepContext, UserEvent
  timing.py                  # precise_sleep + overrun warning
  callbacks.py
  validator.py               # RuntimeValidator
  robot_proxy.py             # ThreadSafeRobotProxy

physicalai/cli/                # in the runtime distribution (physicalai/src/physicalai/)
  main.py                      # console-script entry; jsonargparse dispatcher
  run.py                       # 'physicalai run' subcommand (registers via entry point)
  serve.py                     # 'physicalai serve' subcommand, Phase 5 (registers via entry point)
  # Training subcommands (fit/validate/test/predict/benchmark) are NOT here.
  # They live in library/src/physicalai/{train,benchmark}/cli.py and are
  # discovered via the 'physicalai.cli.subcommands' entry-point group only
  # when the physicalai-train distribution is installed.
```

Studio strategies and other consumer-specific extensions live in their own packages, not in `physicalai`.

---

## 22. References

- `robot_stack_vision.md` — broader stack architecture and naming.
- PhysicalAI design docs: `physicalai/docs/design/components/inferencekit.md`, `physicalai/docs/design/integrations/lerobot.md`.
- PhysicalAI implementation: `physical-ai-studio/library/src/physicalai/inference/`.
- Application implementation: `application/backend/src/workers/`, `application/backend/src/control/`.
- RTC paper: Black, Galliker, Levine. *Real-Time Execution of Action Chunking Flow Policies.* arXiv:2506.07339.
- LeRobot rollout implementation: `lerobot/src/lerobot/rollout/`.
- LeRobot RTC implementation for math cross-checking: `lerobot/src/lerobot/policies/rtc/modeling_rtc.py`.
