# Robot Runtime Architecture

This document defines the production **single-rate robot runtime** for PhysicalAI: the loop that drives one robot at a fixed FPS using one decision-making controller. It is the architecture behind today's policy rollout, teleoperation, recording, HIL, and DAgger workflows, including the existing Studio `RobotControlWorker`.

It supersedes [`policy_runtime_design.md`](./policy_runtime_design.md), which now redirects here. Remote inference details live in [`policy_server_design.md`](./policy_server_design.md). The forward-compatible action/observation evolution lives in [`../robot-interface.md`](../robot-interface.md). Multi-rate, multi-system autonomy for humanoids and mobile manipulators is intentionally **out of scope**; that is the subject of [`composite_robot_architecture.md`](./composite_robot_architecture.md).

The core question:

```text
Is PolicyRuntime the top-level runtime, or is it one controller inside a larger RobotRuntime?
```

Answer:

```text
RobotRuntime      owns the robot loop and lifecycle
Controller        chooses the next RobotAction from an observation
PolicyController  adapts InferenceModel + InferenceExecution + ActionQueue into a Controller
PolicyRuntime     convenience factory that returns a RobotRuntime + PolicyController
```

`PolicyRuntime` remains the simple front door for policy-only deployment. The actual architecture is `RobotRuntime + Controller`.

---

## 1. Goal And Non-Goals

### Goal

Provide a small, predictable, synchronous control loop that:

1. Drives one `Robot` at a fixed FPS.
2. Reads observation data from the robot and any external cameras.
3. Asks a `Controller` for the next `RobotAction`.
4. Filters that action through an optional `SafetyLayer`.
5. Sends the action to the robot.
6. Notifies callbacks for side effects (recording, telemetry, UI events).
7. Handles transient and fatal errors deterministically.

This loop must be reusable across:

- Python scripts (`runtime.run(duration_s=60)`)
- CLI (`physicalai run --config so101_act.yaml`)
- The Studio backend (`RobotControlWorker` orchestrates lifecycle, runs `RobotRuntime` inside)

### Non-Goals

The following are explicitly **not** part of this document:

- Multi-system autonomy (perception + world model + planner + locomotion + VLA arbitration). See [Doc B](./composite_robot_architecture.md).
- Multi-rate subsystem scheduling, blackboards, action arbitration across effectors.
- ROS 2 integration, behavior trees, GXF graphs, distributed runtimes.
- A typed `RobotAction` class hierarchy. The current contract is `np.ndarray`; the migration path to `np.ndarray | Mapping[str, Any]` is documented in [`../robot-interface.md`](../robot-interface.md).
- Replacing application orchestration. `RobotRuntime` is the loop, not the application.

---

## 2. Existing Studio Worker Mapping

`RobotRuntime` does not replace `RobotControlWorker`. It is the **reusable inner loop** that Studio's worker should drive.

```text
RobotControlWorker (existing, async)
  owns:
    process/thread lifecycle
    websocket and queue events
    load/unload model commands
    recording start/stop UI events
    teleop <-> policy mode switching
    backend schemas, model worker registry
  drives:
    RobotRuntime (sync, this doc)

RobotRuntime
  owns the observation -> controller -> action -> robot loop

EnvironmentIntegration / Robot adapter
  owns backend-specific robot/camera wiring and async IO
  exposes the synchronous Robot Protocol upward
```

The existing `RobotControlWorker` uses `async/await` for `EnvironmentIntegration.get_observation`, `set_follower_position_from_leader(goal_time)`, and `set_joints_state`. `RobotRuntime` does **not** become async. Async stays at the adapter boundary; see §7 (Async Application Integration).

---

## 3. Core Model

The runtime loop has three primary values:

```text
observation   what the robot/world sensed, plus per-sample runtime fields
Controller    maps Observation to RobotAction
RobotAction   value accepted by Robot.send_action()
```

High-level flow:

```text
RobotRuntime tick:
  observation = robot.get_observation() + cameras + runtime fields
  observation -> callbacks.on_observation
  action      = controller.update(observation)
  action      = callbacks.before_send_action(action, observation)
  action      = safety.filter(action, observation)   # if configured
  robot.send_action(action)
  callbacks.on_action_sent(action, observation)
  sleep_until_next_tick()
```

Separation of concerns:

```text
RobotRuntime  owns when the loop runs
Controller    owns what action to choose
Robot         owns how to talk to hardware
```

---

## 4. Observation And RobotAction

### Observation

For this runtime architecture, observation is intentionally kept lightweight:

```python
Observation = Mapping[str, Any]
```

Typical keys include `state`, `images`, `task`, `action`, `timestamp`, and runtime-added metadata. Different controllers may consume different subsets.

This document does not require a specific observation class. If the broader runtime later standardizes on a dataclass from [`../observation.md`](../observation.md), that is an implementation choice layered under the same architecture.

### RobotAction (current truth)

The current `Robot` protocol in [`physicalai/robot/interface.py`](../../../../src/physicalai/robot/interface.py) accepts `np.ndarray`:

```python
def send_action(self, action: np.ndarray) -> None: ...
```

This is sufficient for SO-101, Trossen, and any single-arm or single-flat-vector robot — the full set of robots that this runtime targets today.

### RobotAction (forward path)

For composite robots (humanoids, mobile manipulators) a flat ndarray cannot express base / arms / hands / head with mixed control modes. The forward-compatible signature is:

```python
RobotAction = np.ndarray | Mapping[str, Any]
```

`Mapping[str, Any]` is **not** required by this runtime. It is the migration path documented in [`../robot-interface.md`](../robot-interface.md) and consumed by [`composite_robot_architecture.md`](./composite_robot_architecture.md). Single-robot drivers could continue to accept raw `np.ndarray`.

### PolicyAction vs RobotAction

```text
PolicyAction   model output, often normalized and model-specific
RobotAction    robot-facing action accepted by Robot.send_action()
```

`PolicyController` may use an optional `ActionMapper` to convert `PolicyAction` to `RobotAction`. Most existing policies (ACT, Pi0, SmolVLA on SO-101) need no mapper.

---

## 5. PolicyController And Inference

Policy deployment is **one** `Controller` implementation. The model-side abstractions are factored out so they can also run in benchmarks and remote servers.

### Component Ownership

| Component           | Owns                                                                                      | Does NOT own                                |
| ------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------- |
| `InferenceModel`    | load model, preprocess input, run backend, return one action or one chunk                 | robot timing, action queue, callbacks       |
| `InferenceRunner`   | policy computation strategy (single pass, flow matching, temporal ensemble)               | runtime queueing, async scheduling          |
| `ActionChunkCursor` | internal helper: hold one chunk, pop one action at a time                                 | refill policy, smoothing, RTC               |
| `InferenceExecution`| **when and where** inference runs (sync / thread / process / remote)                      | queueing policy, robot IO                   |
| `ActionQueue`       | store chunks, merge chunks, smooth boundaries, pop one action per tick                    | model inference, robot IO                   |
| `PolicyController`  | adapter from the above into a `Controller`                                                | robot lifecycle, FPS, callbacks, safety     |
| `RobotRuntime`      | observation read, controller call, callbacks, safety, dispatch, timing, error handling   | policy math, model loading                  |

### `select_action()` vs `predict_action_chunk()`

`InferenceModel` exposes both, with shape-stable contracts that work for any runner:

```python
action = model.select_action(observation)        # one action now
chunk  = model.predict_action_chunk(observation) # full chunk, no consumption
```

For chunk-producing policies, `select_action()` uses an internal `ActionChunkCursor`:

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(obs))
return cursor.pop()
```

For single-pass runners, `predict_action_chunk()` wraps the single action as a `(1, D)` chunk. This keeps `PolicyController`, `ActionQueue`, `Benchmark`, and `PolicyServer` from branching on runner type.

`InferenceModel` and `InferenceRunner` must not import `ActionQueue`. If both layers need pop-from-chunk mechanics, they share `ActionChunkCursor`, not `ActionQueue`.

### InferenceExecution Modes

```python
class InferenceExecution(Protocol):
    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None: ...
    def maybe_request(self, observation: Mapping[str, Any]) -> None: ...
    def warmup(self, sample_observation: Mapping[str, Any], n: int = 2) -> None: ...
    def stop(self) -> None: ...
```

| Implementation                                  | Where inference runs | Use case                              |
| ----------------------------------------------- | -------------------- | ------------------------------------- |
| `SyncInferenceExecution(mode="single_action")`  | runtime thread       | simple policies                       |
| `SyncInferenceExecution(mode="chunk")`          | runtime thread       | chunk policies, no background worker  |
| `AsyncInferenceExecution(transport="thread")`   | worker thread        | avoid blocking the control loop       |
| `AsyncInferenceExecution(transport="process")`  | worker process       | Studio-style model worker             |
| `RemoteExecution`                               | remote server        | robot host without policy weights/GPU |

`InferenceExecution` is the **only** sync/async boundary the runtime needs. The `RobotRuntime` loop and the `Controller.update()` call stay synchronous regardless of which execution is chosen. See [`policy_server_design.md`](./policy_server_design.md) for `RemoteExecution` and `PolicyServer` details.

### PolicyController

```python
class PolicyController:
    def __init__(
        self,
        model: InferenceModel,
        execution: InferenceExecution,
        action_queue: ActionQueue | None = None,
        fallback: FallbackAction | None = None,
        action_mapper: ActionMapper | None = None,
    ): ...

    def start(self) -> None:
        self.execution.start(self.action_queue, self.model)

    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        self.execution.maybe_request(observation)
        policy_action = self.action_queue.pop_or_none()
        if policy_action is None:
            return self.fallback.action(observation)
        if self.action_mapper is not None:
            return self.action_mapper.to_robot_action(policy_action, observation)
        return policy_action

    def stop(self) -> None:
        self.execution.stop()
        self.model.close()
```

This is the existing `PolicyRuntime` policy logic, but without robot IO ownership.

### ActionQueue

```python
queue = ActionQueue(
    smoother=LerpChunkSmoother(duration_frames=10),
    merger=ReplaceMerger(),
)
queue.push_chunk(chunk)
action = queue.pop_or_none()
```

`ActionQueue` owns runtime action buffering: refill thresholds, cross-chunk smoothing, late results, RTC overlap state. RTC uses a specialized merger:

```python
queue = ActionQueue(merger=RTCQueueMerger())
```

---

## 6. Workflows

All workflows reuse the same `RobotRuntime` loop. Only the `Controller` and `Callback` selection changes.

### 6.1 Policy Rollout

```text
RobotRuntime
  controller = PolicyController(model, execution, action_queue)
```

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=SyncInferenceExecution(mode="chunk"),
    fps=30,
)
runtime.run(duration_s=60)
```

### 6.2 Teleoperation

Teleop is a **source of robot actions**, not a callback. This avoids the dummy-policy hack for data collection.

```text
RobotRuntime
  controller = TeleopController
  callbacks  = [RecordingCallback]
```

```python
class TeleopController:
    def __init__(self, read_input, to_robot_action): ...

    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        teleop_input = self.read_input()
        return self.to_robot_action(teleop_input, observation)
```

The reader can be a keyboard, gamepad, SpaceMouse, leader arm, VR controller, or UI stream. Promote teleop input or mapping to named interfaces only after two or more teleop implementations need the same reusable shape.

### 6.3 Recording

Recording is a **callback** because it observes the loop. It must not arbitrate actions.

```python
class RecordingCallback(RuntimeCallback):
    def on_episode_start(self, episode, step):
        recorder.start_episode(episode_id=episode.id)
    def on_observation(self, obs, step):
        recorder.write_observation(t=step.t, observation=obs)
    def before_send_action(self, action, step):
        recorder.write_policy_action(t=step.t, action=action)
        return action
    def on_action_sent(self, action, step):
        recorder.write_sent_action(t=step.t, action=action)
    def on_episode_end(self, episode, step):
        recorder.end_episode()
```

### 6.4 HIL (Human-In-The-Loop)

HIL is **arbitration between two action sources** and is a controller, not a callback.

```python
class HILController:
    def __init__(self, policy: PolicyController, expert: TeleopController, mode_source): ...

    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        policy_action = self.policy.update(observation)
        expert_action = self.expert.peek_or_read(observation)
        return arbitrate(policy_action, expert_action, self.mode_source.current())
```

### 6.5 DAgger

DAgger writes both actions to a dataset and chooses one based on a beta schedule. Also a controller.

```python
class DAggerController:
    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        policy_action = self.policy.update(observation)
        expert_action = self.expert.read(observation)
        self.dataset_writer.write(observation, policy_action, expert_action)
        if random() < self.beta_schedule(observation):
            return expert_action
        return policy_action
```

### 6.6 Workflow Mapping

| Workflow              | Controller         | Callbacks                        |
| --------------------- | ------------------ | -------------------------------- |
| policy rollout        | `PolicyController` | optional telemetry/recording     |
| teleop data collection| `TeleopController` | `RecordingCallback`              |
| recorded rollout      | `PolicyController` | `RecordingCallback`              |
| highlight capture     | any                | `HighlightCallback`              |
| HIL                   | `HILController`    | optional recording/telemetry     |
| DAgger                | `DAggerController` | optional metrics/recording       |
| scripted collection   | `ScriptedController`| `RecordingCallback`             |

Multi-system autonomy (VLA + locomotion + perception + planner) is **not** a controller in this doc. It is an `AutonomyController` defined in [`composite_robot_architecture.md`](./composite_robot_architecture.md) that satisfies the same `Controller` protocol.

---

## 7. Async Application Integration?

The existing `RobotControlWorker` is fully async. `RobotRuntime` is sync. The two coexist via the `Robot` adapter boundary.

### Rule

```text
async stays at application/adapter boundaries
Controller.update() stays synchronous and non-blocking
```

### Two Acceptable Patterns

**Pattern 1: Run RobotRuntime in a worker thread/process.**

```text
RobotControlWorker (async)
  websocket / queue events
  load/unload commands
  spawns thread:
    RobotRuntime.run(...)            # sync loop in its own thread
  uses thread-safe runtime methods to inject mid-loop commands (see §9)
```

**Pattern 2: Wrap async robot/environment IO behind a synchronous adapter.**

```text
RobotControlWorker (async)
  drives an AsyncEnvironmentRobot adapter:
    background task continuously calls EnvironmentIntegration.get_observation
    caches latest observation in a thread-safe slot
    send_action enqueues to a bounded queue drained by an async worker
  exposes synchronous Robot Protocol to RobotRuntime
```

The first pattern is recommended for the initial migration because it preserves Studio's existing async event handling unchanged. The second pattern is more invasive but allows native single-process operation.

For the leader/follower `goal_time` budget used by Trossen teleop today, the adapter is responsible for translating `send_action(np.ndarray)` into the timed async call. `RobotRuntime` does not need to know.

### What `Controller.update()` MUST NOT do

- Block on network IO without a timeout
- Block on device reads without a cached fallback
- Call `await` (it is sync)
- Hold the GIL for longer than one tick budget

Controllers that need slow IO (device reads, network calls) must buffer or poll internally. A teleop controller should read device state in a background thread and return the latest cached value from `update()`. This keeps the runtime tick predictable and decouples the loop rate from device latency.

---

## 8. Interfaces

### 8.1 RobotRuntime

```python
class RobotRuntime:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        fps: float,
        cameras: Mapping[str, Camera] | None = None,
        callbacks: Sequence[RuntimeCallback] = (),
        safety: SafetyLayer | None = None,
        return_to_home: bool = False,
    ): ...

    def run(self, *, duration_s: float | None = None) -> None: ...
    def stop(self) -> None: ...
    def swap_controller(self, controller: Controller) -> None: ...

    @classmethod
    def from_config(cls, path: str | Path) -> "RobotRuntime": ...
```

Loop body:

```python
while running:
    try:
        observation = robot.get_observation()
    except Exception:
        break  # robot IO failure -> safe shutdown

    observation = merge_camera_observations(observation, cameras)
    observation = add_runtime_observation_fields(
        observation, timestamp=clock.now(),
        frame_index=frame_index, episode_index=episode_index,
    )
    callbacks.on_observation(observation)

    try:
        action = controller.update(observation)
    except Exception as e:
        callbacks.on_error(e, observation)
        action = last_safe_action  # hold position; transient failure

    action = callbacks.before_send_action(action, observation)

    if safety is not None:
        try:
            action = safety.filter(action, observation)
        except SafetyViolationError:
            break  # safety cannot make action safe -> safe shutdown

    try:
        robot.send_action(action)
    except Exception:
        break  # robot IO failure -> safe shutdown

    last_safe_action = action
    callbacks.on_action_sent(action, observation)
    sleep_until_next_tick()

# safe shutdown
controller.stop()
callbacks.on_stop()
if return_to_home:
    robot.go_to_home()
robot.disconnect()
```

### 8.2 Controller

```python
class Controller(Protocol):
    def start(self) -> None: ...
    def update(self, observation: Mapping[str, Any]) -> RobotAction: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
```

The contract is intentionally small. A controller may be a policy, teleop device, planner, behavior tree, hierarchical autonomy stack, or a composition of other controllers (see Doc B's `AutonomyController`).

`update()` is synchronous and must not block.

### 8.3 RuntimeCallback

```python
class RuntimeCallback(Protocol):
    def on_start(self) -> None: ...
    def on_observation(self, observation: Mapping[str, Any]) -> None: ...
    def before_send_action(self, action: RobotAction, observation: Mapping[str, Any]) -> RobotAction: ...
    def on_action_sent(self, action: RobotAction, observation: Mapping[str, Any]) -> None: ...
    def on_error(self, error: Exception, observation: Mapping[str, Any] | None = None) -> None: ...
    def on_user_event(self, event: UserEvent, observation: Mapping[str, Any] | None = None) -> None: ...
    def on_stop(self) -> None: ...
```

Callbacks are for **side effects and instrumentation**: recording, highlight capture, telemetry, logging, UI updates, metrics, watchdog notifications.

Callbacks are **not** the abstraction for action arbitration. If a component regularly decides which action to send, make it a `Controller`.

`before_send_action` may modify the action (for example, an HIL UI override), but the safety layer always runs after callbacks.

### 8.4 SafetyLayer

```python
class SafetyLayer(Protocol):
    def filter(self, action: RobotAction, observation: Mapping[str, Any]) -> RobotAction: ...

class SafetyViolationError(Exception):
    """Raised by SafetyLayer when an action cannot be made safe."""
```

Safety is separate from callbacks because it is part of the action path with explicit ordering: **after** `callbacks.before_send_action`, **before** `robot.send_action`. Raising `SafetyViolationError` triggers safe shutdown.

### 8.5 Error Handling

| Source                                     | Behavior                                                        |
| ------------------------------------------ | --------------------------------------------------------------- |
| `controller.update()` raises               | log, `on_error`, hold last safe action, continue loop           |
| `robot.send_action()` raises               | log, `on_error`, stop loop, safe shutdown                       |
| `SafetyLayer.filter()` raises `SafetyViolationError` | log, `on_error`, stop loop, safe shutdown             |
| `SafetyLayer.filter()` raises other        | treat as controller error: log, hold position, continue         |
| `robot.get_observation()` raises           | log, stop loop, safe shutdown                                   |

Safe shutdown:

```text
1. controller.stop()
2. callbacks.on_stop()
3. robot.go_to_home() if return_to_home
4. robot.disconnect()
```

Controller errors do not stop the loop by default because transient inference failures (a model worker timeout) should not crash a running robot. Robot IO errors stop the loop because they indicate a hardware failure that retries cannot recover.

---

## 9. Mid-Loop Commands

`RobotRuntime.run()` is a blocking call. The application needs a way to inject commands without an internal event/command queue.

| Operation             | Mechanism                                                       |
| --------------------- | --------------------------------------------------------------- |
| stop the loop         | `runtime.stop()` sets an atomic flag checked each tick          |
| swap controller       | `runtime.swap_controller(c)` replaces the reference under a lock|
| change task           | controller-level method, not a runtime concern                  |
| start/stop recording  | callback-level method, not a runtime concern                    |
| load model            | application creates a new `PolicyController` and swaps it in    |

Thread-safe attribute swaps and atomic flags are sufficient. Application orchestration logic stays in the application.

---

## 10. Robot IO Data Plane

Some robots need hardware IO faster than the runtime loop. A Trossen leader/follower pair may need a 100 Hz teleoperation loop while inference and UI run at 30 Hz.

The `Robot` adapter hides this:

```text
RobotRuntime (30 Hz)
  robot.get_observation()   reads latest state from internal buffer
  robot.send_action(action) writes latest action to internal buffer
        |
        v
Robot adapter (internal)
  high-rate thread/process (100 Hz)
  reads/writes hardware at device rate
  bridges via shared buffers
```

`RobotRuntime` does not need to know the transport. The `Robot` interface is the boundary. Do not add transport, data plane, or control port abstractions to the framework until at least two robot integrations need the same reusable implementation.

---

## 11. PolicyRuntime: Convenience Factory

`PolicyRuntime` is **not** a second runtime. It is a one-line factory over `RobotRuntime + PolicyController`.

```python
def PolicyRuntime(
    *,
    robot: Robot,
    model: InferenceModel,
    execution: InferenceExecution,
    fps: float,
    action_queue: ActionQueue | None = None,
    **kwargs,
) -> RobotRuntime:
    return RobotRuntime(
        robot=robot,
        controller=PolicyController(
            model=model,
            execution=execution,
            action_queue=action_queue,
        ),
        fps=fps,
        **kwargs,
    )
```

Simple policy API stays:

```python
runtime = PolicyRuntime(robot=robot, model=model, execution=SyncInferenceExecution(mode="chunk"), fps=30)
runtime.run(duration_s=60)
```

General API also available:

```python
runtime = RobotRuntime(
    robot=robot,
    controller=TeleopController(read_input=gamepad.read, to_robot_action=mapper),
    callbacks=[RecordingCallback()],
    fps=30,
)
runtime.run()
```

---

## 12. Config And CLI

### Config Examples

Policy rollout:

```yaml
runtime:
  class_path: physicalai.runtime.RobotRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args: { port: /dev/ttyACM0 }
    controller:
      class_path: physicalai.runtime.PolicyController
      init_args:
        model:
          class_path: physicalai.inference.InferenceModel
          init_args: { path: ./exports/act_policy }
        execution:
          class_path: physicalai.runtime.SyncInferenceExecution
          init_args: { mode: chunk }
```

Teleop recording:

```yaml
runtime:
  class_path: physicalai.runtime.RobotRuntime
  init_args:
    fps: 30
    robot:
      class_path: physicalai.robot.so101.SO101
      init_args: { port: /dev/ttyACM0 }
    controller:
      class_path: physicalai.runtime.TeleopController
      init_args:
        device:
          class_path: physicalai.teleop.Gamepad
    callbacks:
      - class_path: physicalai.data.RecordingCallback
        init_args: { output_dir: ./datasets/teleop_run_001 }
```

HIL with async process inference:

```yaml
runtime:
  class_path: physicalai.runtime.RobotRuntime
  init_args:
    fps: 30
    robot: { class_path: physicalai.robot.so101.SO101 }
    controller:
      class_path: physicalai.runtime.HILController
      init_args:
        policy:
          class_path: physicalai.runtime.PolicyController
          init_args:
            model:
              class_path: physicalai.inference.InferenceModel
              init_args: { path: ./exports/act_policy }
            execution:
              class_path: physicalai.runtime.AsyncInferenceExecution
              init_args: { transport: process }
        expert:
          class_path: physicalai.runtime.TeleopController
```

### CLI

Runtime CLI lives in the `physicalai` distribution, not in the training distribution:

```toml
[project.scripts]
physicalai = "physicalai.cli.main:main"

[project.entry-points."physicalai.cli.subcommands"]
run   = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"
```

Training commands plug in from the training distribution:

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit       = "physicalai.train.cli:register_fit"
validate  = "physicalai.train.cli:register_validate"
test      = "physicalai.train.cli:register_test"
predict   = "physicalai.train.cli:register_predict"
benchmark = "physicalai.benchmark.cli:register"
```

This keeps `physicalai run` usable on inference hosts without Torch or Lightning.

```bash
physicalai run --config so101_act.yaml --duration-s 60
```

---

## 13. Benchmarking vs Runtime

`physicalai.benchmark` evaluates policies across gyms/tasks; it is **not** a second runtime. It owns episode aggregation, success rate, reward, episode length, average FPS, optional video, and JSON/CSV export. It calls `model.select_action()` directly.

| Concern                                 | `Benchmark` | `RobotRuntime` |
| --------------------------------------- | ----------- | -------------- |
| Task suite / gym orchestration          | yes         | no             |
| Episode aggregation, success metrics    | yes         | no             |
| Video / result export                   | yes         | no             |
| Robot/camera connection lifecycle       | no          | yes            |
| FPS-controlled robot loop               | no          | yes            |
| Runtime-owned action queue              | no          | yes            |
| Async/process/remote inference scheduling | no        | yes            |

Do not make `Benchmark` a second implementation of the production robot loop.

---

## 14. Naming

| Name                  | Recommendation                | Reason                                                        |
| --------------------- | ----------------------------- | ------------------------------------------------------------- |
| `RobotRuntime`        | preferred top-level name      | owns the robot loop, not only policy inference                |
| `Controller`          | preferred action-selection abstraction | covers policy, teleop, planner, HIL, DAgger, autonomy |
| `PolicyController`    | preferred policy adapter      | wraps `InferenceModel`, `InferenceExecution`, `ActionQueue`   |
| `PolicyRuntime`       | convenience factory           | one-line API for policy-only deployment                       |
| `InferenceExecution`  | preferred over `Execution`    | precise; the only sync/async boundary                         |
| `RobotAction`         | preferred runtime output name | matches `Robot.send_action()`                                 |
| `ActionSource`        | avoid                         | implies emit-only without lifecycle                           |
| `RobotCommand`        | defer                         | useful later if actions become typed command objects          |
| `ControlRuntime`      | acceptable but broader        | could describe non-robot control loops                        |

---

## 15. What This Doc Does NOT Define

The following are intentionally out of scope and live in [`composite_robot_architecture.md`](./composite_robot_architecture.md):

- `AutonomyController` (multi-system controller that implements `Controller`)
- `RuntimeSubsystem` protocol and lifecycle
- `Blackboard`, freshness/TTL/deadline semantics
- Multi-rate scheduling, degrade behavior
- `ActionArbiter`, effector-scoped actions
- ROS 2 / GXF / behavior tree interop

The following are intentionally deferred until a concrete consumer needs them:

- `HILController`, `DAggerController` (designs above; implementations on demand)
- Typed `RobotAction` classes (see [`../robot-interface.md`](../robot-interface.md))
- Separate runtime context/tick objects
- `RobotTransport` / `RobotIODataPlane` / `RobotControlPort` abstractions, until two robot integrations need them
- Advanced event bus

---

## 16. Implementation Phases

1. ✅ `InferenceModel.predict_action_chunk()` and `close()`.
2. ✅ `RobotRuntime`, `Controller`, `PolicyController`, `RuntimeCallback`, `ActionQueue`, `SyncInferenceExecution`, `PolicyRuntime` factory, validation.
3. ✅ `AsyncInferenceExecution(transport=thread)` + `FallbackAction` + `runtime.warmup()`. Validated on SO-101 + MolmoAct2 at 2 Hz. `transport=process` deferred (YAGNI).
4. RTC delay tracking and `RTCQueueMerger`.
5. `RemoteExecution`, `PolicyServer`, `physicalai serve` (see [`policy_server_design.md`](./policy_server_design.md)).
6. `SafetyLayer` and `SafetyViolationError`, when a first consumer needs action filtering.
7. Doc B implementation (`AutonomyController`, `RuntimeSubsystem`, `Blackboard`, `ActionArbiter`) once a humanoid integration is concrete.

---

## 17. Decision Summary

```text
robot loop ownership      RobotRuntime
action selection          Controller
policy inference          PolicyController + InferenceExecution
sync/async boundary       InferenceExecution (sync | async-thread | async-process | remote)
per-sample data           observation mapping with conventional keys
side effects              RuntimeCallback (no arbitration)
safety enforcement        SafetyLayer (after callbacks, before send_action)
error recovery            controller errors hold position; robot/safety errors stop loop
application commands      thread-safe methods on runtime, controller, and callbacks
async studio worker       wraps RobotRuntime in a thread, OR async behind Robot adapter
multi-system autonomy     AutonomyController in composite_robot_architecture.md
```

This design keeps the policy deployment path simple while leaving room for full robot control systems without turning callbacks into an implicit control architecture and without forcing the loop to become async.
