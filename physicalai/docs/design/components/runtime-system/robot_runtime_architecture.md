# Robot Runtime Architecture

This document defines the production single-rate robot runtime for PhysicalAI: one robot, one fixed-FPS loop, and one active controller. It is the architecture for policy rollout, teleoperation, recording, HIL, DAgger, and the existing Studio `RobotControlWorker` integration.

It supersedes [`policy_runtime_design.md`](./policy_runtime_design.md), which now redirects here. Remote inference details live in [`policy_server_design.md`](./policy_server_design.md). Forward-compatible action and observation evolution lives in [`../robot-interface.md`](../robot-interface.md). Multi-rate, multi-system autonomy is out of scope here and belongs in [`composite_robot_architecture.md`](./composite_robot_architecture.md).

## Quick Navigation

- [At a Glance](#at-a-glance)
- [PolicyRuntime vs RobotRuntime](#policyruntime-vs-robotruntime)
- [Scope](#scope)
- [Studio Mapping](#studio-mapping)
- [Runtime Loop](#runtime-loop)
- [Core Data Model](#core-data-model)
- [Controllers and Inference](#controllers-and-inference)
- [Workflow Patterns](#workflow-patterns)
- [Async Integration](#async-integration)
- [Interfaces](#interfaces)
- [Errors and Shutdown](#errors-and-shutdown)
- [Mid-Loop Commands](#mid-loop-commands)
- [Robot IO Data Plane](#robot-io-data-plane)
- [PolicyRuntime Factory](#policyruntime-factory)
- [Config and CLI](#config-and-cli)
- [Benchmarking Boundary](#benchmarking-boundary)
- [Naming and Boundaries](#naming-and-boundaries)
- [Implementation Status](#implementation-status)
- [Decision Summary](#decision-summary)

## At a Glance

The main design question is:

```text
Is PolicyRuntime the top-level runtime, or is it one controller inside a larger RobotRuntime?
```

The answer is:

```text
RobotRuntime      owns the robot loop and lifecycle
Controller        chooses the next robot action
PolicyController  adapts inference components into a Controller
PolicyRuntime     is a convenience factory that returns RobotRuntime + PolicyController
```

The actual architecture is `RobotRuntime + Controller`. `PolicyRuntime` remains the simple entry point for policy-only deployment.

## PolicyRuntime vs RobotRuntime

`PolicyRuntime` is the policy-first API. `RobotRuntime` is the reusable loop beneath it.

| Concern | Current `PolicyRuntime` shape | `RobotRuntime` shape |
| --- | --- | --- |
| top-level purpose | run a policy on a robot | run any controller on a robot |
| action source | policy inference | policy, teleop, HIL, DAgger, scripted control, autonomy |
| model ownership | direct runtime concern | owned by `PolicyController` |
| inference scheduling | direct runtime concern | owned by `InferenceExecution` |
| recording and telemetry | side behavior around policy rollout | callbacks around any controller |
| Studio integration | policy-oriented worker behavior | Studio orchestrates modes, runtime executes loop |

The migration path is additive:

```python
# policy-only API remains simple
runtime = PolicyRuntime(robot=robot, model=model, execution=execution, fps=30)

# general API exposes the underlying architecture
runtime = RobotRuntime(robot=robot, controller=controller, fps=30)
```

Do not create two production loops. `PolicyRuntime` should remain a factory over `RobotRuntime + PolicyController`.

## Scope

### Goals

`RobotRuntime` provides a small, predictable loop that:

1. Drives one `Robot` at a fixed FPS.
2. Reads robot state and optional camera data.
3. Calls a `Controller` for the next action.
4. Applies callbacks and an optional `SafetyLayer`.
5. Sends the action to the robot.
6. Emits side effects for recording, telemetry, and UI.
7. Handles transient and fatal errors deterministically.

It is designed to be reused from scripts, CLI entry points, and Studio backend workers.

### Out of Scope

This document does not define:

- Multi-system autonomy, planners, world models, and locomotion orchestration. See [`composite_robot_architecture.md`](./composite_robot_architecture.md).
- Multi-rate subsystem scheduling, blackboards, and effector arbitration.
- ROS 2, behavior trees, GXF graphs, or distributed runtime frameworks.
- A typed `RobotAction` class hierarchy.
- Full application orchestration. `RobotRuntime` is the loop, not the application.

## Studio Mapping

`RobotRuntime` does not replace `RobotControlWorker`. It becomes the reusable synchronous loop inside it.

```text
RobotControlWorker (existing, async)
  owns:
    process or thread lifecycle
    websocket and queue events
    model load and unload commands
    recording UI commands
    teleop and policy mode switching
    backend schemas and worker registry
  drives:
    RobotRuntime (sync)

RobotRuntime
  owns:
    observation -> controller -> action -> robot loop

EnvironmentIntegration / Robot adapter
  owns:
    backend-specific async IO
    robot and camera wiring
  exposes:
    synchronous Robot protocol to RobotRuntime
```

Async remains at the adapter boundary. `RobotRuntime` stays synchronous.

## Runtime Loop

The runtime has one job: read an observation, ask for an action, send it, and repeat on a fixed schedule.

```text
tick:
  observation = robot.get_observation() + cameras + runtime fields
  callbacks.on_observation(observation)
  action = controller.update(observation)
  action = callbacks.before_send_action(action, observation)
  action = safety.filter(action, observation)      # optional
  robot.send_action(action)
  callbacks.on_action_sent(action, observation)
  sleep_until_next_tick()
```

Ownership is intentionally simple:

```text
RobotRuntime  decides when the loop runs
Controller    decides what action to take
Robot         decides how hardware IO happens
```

### Reference Loop

```python
while running:
    try:
        observation = robot.get_observation()
    except Exception:
        break

    observation = merge_camera_observations(observation, cameras)
    observation = add_runtime_observation_fields(
        observation,
        timestamp=clock.now(),
        frame_index=frame_index,
        episode_index=episode_index,
    )
    callbacks.on_observation(observation)

    try:
        action = controller.update(observation)
    except Exception as error:
        callbacks.on_error(error, observation)
        action = last_safe_action

    action = callbacks.before_send_action(action, observation)

    if safety is not None:
        try:
            action = safety.filter(action, observation)
        except SafetyViolationError:
            break

    try:
        robot.send_action(action)
    except Exception as error:
        callbacks.on_error(error, observation)
        break

    last_safe_action = action
    callbacks.on_action_sent(action, observation)
    sleep_until_next_tick()

controller.stop()
callbacks.on_stop()
if return_to_home:
    robot.go_to_home()
robot.disconnect()
```

## Core Data Model

### Observation

Observation stays lightweight in this architecture:

```python
Observation = Mapping[str, Any]
```

Typical keys include:

```python
{
    "state": ...,
    "images": ...,
    "task": ...,
    "action": ...,
    "timestamp": ...,
    "frame_index": ...,
}
```

This document does not require a dedicated observation class.

### Action Boundary

A controller returns the value that the runtime passes to `Robot.send_action()`.

```python
action = controller.update(observation)
robot.send_action(action)
```

Today this is usually an `np.ndarray` for single-arm or flat-vector robots. More complex action schemas are covered in [`../robot-interface.md`](../robot-interface.md) and [`composite_robot_architecture.md`](./composite_robot_architecture.md).

## Controllers and Inference

Policy deployment is one `Controller` implementation, not a second runtime.

### Responsibility Split

| Component | Owns | Does not own |
| --- | --- | --- |
| `InferenceModel` | model loading, preprocessing, backend execution, single action or chunk output | robot timing, callbacks, robot IO |
| `InferenceRunner` | policy computation strategy | runtime scheduling, action buffering |
| `ActionChunkCursor` | hold one chunk and pop from it | refilling, smoothing, queue policy |
| `InferenceExecution` | when and where inference runs | robot IO, action queue policy |
| `ActionQueue` | buffering, merging, smoothing, one-action-per-tick consumption | model execution |
| `PolicyController` | adapts inference stack into `Controller` | runtime lifecycle, FPS, safety |
| `RobotRuntime` | observation read, callbacks, safety, dispatch, timing, shutdown | policy math and model loading |

### Single Action vs Chunked Action

`InferenceModel` exposes both contracts:

```python
action = model.select_action(observation)
chunk = model.predict_action_chunk(observation)
```

For chunk-producing policies, `select_action()` can be implemented through `ActionChunkCursor`:

```python
if cursor.empty():
    cursor.push_chunk(model.predict_action_chunk(observation))
return cursor.pop()
```

For single-pass policies, `predict_action_chunk()` returns a one-step chunk:

```python
return action[None, :]
```

This keeps downstream consumers shape-stable.

### InferenceExecution

`InferenceExecution` is the only policy-side sync, async, process, or remote boundary the runtime needs. The `RobotRuntime` tick still calls `controller.update(observation)` synchronously; async inference is hidden behind `maybe_request()` and `ActionQueue`.

```python
class InferenceExecution(Protocol):
    def start(self, action_queue: ActionQueue, model: InferenceModel) -> None: ...
    def maybe_request(self, observation: Mapping[str, Any]) -> None: ...
    def warmup(self, sample_observation: Mapping[str, Any], n: int = 2) -> None: ...
    def stop(self) -> None: ...
```

| Implementation | Where inference runs | Typical use |
| --- | --- | --- |
| `SyncInferenceExecution(mode="single_action")` | runtime thread | simple policies |
| `SyncInferenceExecution(mode="chunk")` | runtime thread | chunk policies without background workers |
| `AsyncInferenceExecution(transport="thread")` | worker thread | avoid blocking the control loop |
| `AsyncInferenceExecution(transport="process")` | worker process | Studio-style model worker |
| `RemoteExecution` | remote server | robot host without local model weights or GPU |

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

### ActionQueue

```python
queue = ActionQueue(
    smoother=LerpChunkSmoother(duration_frames=10),
    merger=ReplaceMerger(),
)

queue.push_chunk(chunk)
action = queue.pop_or_none()
```

RTC uses a different merger, not a different runtime:

```python
queue = ActionQueue(merger=RTCQueueMerger())
```

## Workflow Patterns

All workflows reuse the same runtime loop. The main variation is the selected controller and callbacks.

### Policy Rollout

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=SyncInferenceExecution(mode="chunk"),
    fps=30,
)
runtime.run(duration_s=60)
```

### Teleoperation

Teleop is a controller because it produces actions.

```python
class TeleopController:
    def __init__(self, read_input, to_robot_action): ...

    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        teleop_input = self.read_input()
        return self.to_robot_action(teleop_input, observation)
```

### Recording

Recording is a callback because it observes the loop.

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

### Human-In-The-Loop

HIL is a controller because it arbitrates between action sources.

```python
class HILController:
    def __init__(self, policy: PolicyController, expert: TeleopController, mode_source): ...

    def update(self, observation: Mapping[str, Any]) -> RobotAction:
        policy_action = self.policy.update(observation)
        expert_action = self.expert.peek_or_read(observation)
        return arbitrate(policy_action, expert_action, self.mode_source.current())
```

### DAgger

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

### Workflow Mapping

| Workflow | Controller | Callbacks |
| --- | --- | --- |
| policy rollout | `PolicyController` | optional telemetry or recording |
| teleop data collection | `TeleopController` | `RecordingCallback` |
| recorded rollout | `PolicyController` | `RecordingCallback` |
| highlight capture | any | `HighlightCallback` |
| HIL | `HILController` | optional recording or telemetry |
| DAgger | `DAggerController` | optional metrics or recording |
| scripted collection | `ScriptedController` | `RecordingCallback` |

Multi-system autonomy is not another special case here. It is a controller defined in [`composite_robot_architecture.md`](./composite_robot_architecture.md) that satisfies the same `Controller` protocol.

## Async Integration

`RobotRuntime` should stay synchronous unless a concrete integration proves otherwise. A fixed-rate control loop is easier to test, profile, and fail safely when each tick is a normal blocking function call with bounded work.

The rule is about the control loop boundary, not about forbidding async work behind components:

```text
application async      Studio events, UI commands, mode orchestration
policy async           InferenceExecution + ActionQueue
robot/env async        Robot adapter boundary
runtime tick           synchronous Controller.update() call with bounded work
```

Studio can remain async. Teleop, HIL, DAgger, recording commands, websocket events, and model loading can all stay in Studio orchestration. The runtime only needs the current controller, the latest observation, and a bounded action path.

Remote inference fits the same model. `RemoteExecution.maybe_request()` submits or polls bounded work, pushes completed chunks into `ActionQueue`, and returns without waiting for the remote server on every tick.

### Pattern 1: Run `RobotRuntime` in a Worker

This is the recommended migration path for Studio.

```text
RobotControlWorker (async)
  handles websocket and queue events
  loads and unloads controllers or models
  owns teleop/HIL/DAgger mode orchestration
  spawns a thread or process
    RobotRuntime.run(...)
```

### Pattern 2: Wrap Async IO Behind a Sync Robot Adapter

```text
RobotControlWorker (async)
  drives an async environment integration
  background task fetches observations
  adapter caches latest observation
  adapter queues outgoing actions
  RobotRuntime sees a synchronous Robot interface
```

Use this pattern only when keeping the runtime in-process is worth the adapter complexity.

### What `Controller.update()` Must Not Do

- Block on unbounded network IO.
- Block on device reads without a cached fallback.
- Call `await`.
- Hold the GIL longer than the tick budget.

Controllers that depend on slow IO should buffer internally and return the latest available value.

## Interfaces

### RobotRuntime

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

### Controller

```python
class Controller(Protocol):
    def start(self) -> None: ...
    def update(self, observation: Mapping[str, Any]) -> RobotAction: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
```

The contract stays small on purpose. A controller may be a policy, teleop source, planner, autonomy stack, or composition of controllers.

### RuntimeCallback

```python
class RuntimeCallback(Protocol):
    def on_start(self) -> None: ...
    def on_observation(self, observation: Mapping[str, Any]) -> None: ...
    def before_send_action(
        self,
        action: RobotAction,
        observation: Mapping[str, Any],
    ) -> RobotAction: ...
    def on_action_sent(self, action: RobotAction, observation: Mapping[str, Any]) -> None: ...
    def on_error(self, error: Exception, observation: Mapping[str, Any] | None = None) -> None: ...
    def on_user_event(self, event: UserEvent, observation: Mapping[str, Any] | None = None) -> None: ...
    def on_stop(self) -> None: ...
```

Callbacks are for side effects and instrumentation, not regular action arbitration.

### SafetyLayer

```python
class SafetyLayer(Protocol):
    def filter(self, action: RobotAction, observation: Mapping[str, Any]) -> RobotAction: ...

class SafetyViolationError(Exception):
    """Raised when an action cannot be made safe."""
```

Ordering is explicit:

```text
controller -> callbacks.before_send_action -> safety.filter -> robot.send_action
```

## Errors and Shutdown

### Error Handling

| Source | Runtime behavior |
| --- | --- |
| `controller.update()` raises | log, call `on_error`, hold last safe action, continue |
| `robot.send_action()` raises | log, call `on_error`, stop loop, safe shutdown |
| `SafetyLayer.filter()` raises `SafetyViolationError` | log, call `on_error`, stop loop, safe shutdown |
| `SafetyLayer.filter()` raises other exception | treat like controller error |
| `robot.get_observation()` raises | log, stop loop, safe shutdown |

Controller errors do not stop the loop by default. Robot IO and hard safety failures do.

### Safe Shutdown Order

```text
1. controller.stop()
2. callbacks.on_stop()
3. robot.go_to_home() if return_to_home
4. robot.disconnect()
```

## Mid-Loop Commands

`RobotRuntime.run()` is blocking, so application-level commands enter through thread-safe runtime, controller, or callback methods.

| Operation | Mechanism |
| --- | --- |
| stop the loop | `runtime.stop()` sets an atomic flag |
| swap controller | `runtime.swap_controller(controller)` under a lock |
| change task | controller-level API |
| start or stop recording | callback-level API |
| load model | application constructs a new `PolicyController` and swaps it in |

The runtime does not need an internal command bus for these cases.

## Robot IO Data Plane

Some robots need higher-rate hardware IO than the runtime loop.

```text
RobotRuntime (30 Hz)
  robot.get_observation()   reads latest buffered state
  robot.send_action(action) writes latest buffered action
        |
        v
Robot adapter (internal)
  high-rate thread or process (for example 100 Hz)
  talks to hardware at device rate
  bridges with shared buffers
```

The `Robot` adapter hides transport details. Do not introduce framework-level transport abstractions until multiple integrations need the same reusable shape.

## PolicyRuntime Factory

`PolicyRuntime` is not a separate runtime. It is a convenience factory.

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

Two equivalent entry points:

```python
runtime = PolicyRuntime(
    robot=robot,
    model=model,
    execution=SyncInferenceExecution(mode="chunk"),
    fps=30,
)
runtime.run(duration_s=60)
```

```python
runtime = RobotRuntime(
    robot=robot,
    controller=TeleopController(read_input=gamepad.read, to_robot_action=mapper),
    callbacks=[RecordingCallback()],
    fps=30,
)
runtime.run()
```

## Config and CLI

### Config Example: Policy Rollout

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

### Config Example: Teleop Recording

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

### Config Example: HIL with Process Inference

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

### CLI Boundary

Runtime commands live in the runtime distribution:

```toml
[project.scripts]
physicalai = "physicalai.cli.main:main"

[project.entry-points."physicalai.cli.subcommands"]
run   = "physicalai.cli.run:register"
serve = "physicalai.cli.serve:register"
```

Training commands come from the training distribution:

```toml
[project.entry-points."physicalai.cli.subcommands"]
fit       = "physicalai.train.cli:register_fit"
validate  = "physicalai.train.cli:register_validate"
test      = "physicalai.train.cli:register_test"
predict   = "physicalai.train.cli:register_predict"
benchmark = "physicalai.benchmark.cli:register"
```

Example runtime invocation:

```bash
physicalai run --config so101_act.yaml --duration-s 60
```

## Benchmarking Boundary

`physicalai.benchmark` is not a second production runtime.

| Concern | `Benchmark` | `RobotRuntime` |
| --- | --- | --- |
| task suite and gym orchestration | yes | no |
| episode aggregation and metrics | yes | no |
| result export and video | yes | no |
| robot and camera lifecycle | no | yes |
| fixed-FPS robot loop | no | yes |
| runtime-owned action queue | no | yes |
| async, process, or remote inference scheduling | no | yes |

`Benchmark` can call `model.select_action()` directly. It should not reimplement the production robot loop.

## Naming and Boundaries

### Recommended Names

| Name | Recommendation | Reason |
| --- | --- | --- |
| `RobotRuntime` | preferred top-level name | it owns the robot loop |
| `Controller` | preferred action-selection abstraction | it covers policy, teleop, HIL, DAgger, and autonomy |
| `PolicyController` | preferred policy adapter | it wraps inference components |
| `PolicyRuntime` | convenience factory | it keeps the policy-only API simple |
| `InferenceExecution` | preferred over `Execution` | it names the sync-async boundary clearly |
| `RobotAction` | preferred runtime output name | it matches `Robot.send_action()` |
| `ActionSource` | avoid | it suggests no lifecycle |
| `RobotCommand` | defer | it may be useful later if actions become typed objects |

### Still Out of Scope

The following remain intentionally outside this document:

- `AutonomyController`, `RuntimeSubsystem`, and `Blackboard`.
- Multi-rate scheduling, degrade behavior, and `ActionArbiter`.
- ROS 2, GXF, and behavior tree interop.
- Framework-level transport abstractions.
- Typed action classes until there is a concrete consumer.

The following are also intentionally deferred until a concrete need appears:

- `HILController` and `DAggerController` implementations.
- Separate runtime context or tick objects.
- An internal event bus.

## Implementation Status

1. `InferenceModel.predict_action_chunk()` and `close()` are implemented.
2. `RobotRuntime`, `Controller`, `PolicyController`, `RuntimeCallback`, `ActionQueue`, `SyncInferenceExecution`, and the `PolicyRuntime` factory are implemented.
3. `AsyncInferenceExecution(transport="thread")`, `FallbackAction`, and `runtime.warmup()` are implemented and validated on SO-101 + MolmoAct2 at 2 Hz.
4. `RTCQueueMerger` and delay tracking are next runtime-side extensions.
5. `RemoteExecution`, `PolicyServer`, and `physicalai serve` are defined in [`policy_server_design.md`](./policy_server_design.md).
6. `SafetyLayer` lands when there is a concrete consumer for action filtering.
7. Composite runtime constructs land with the first concrete humanoid or multi-system integration.

## Decision Summary

```text
robot loop ownership      RobotRuntime
action selection          Controller
policy inference          PolicyController + InferenceExecution
sync or async boundary    InferenceExecution
per-sample data           observation mapping with conventional keys
side effects              RuntimeCallback
safety enforcement        SafetyLayer after callbacks, before send_action
error recovery            controller errors hold position; robot and safety errors stop loop
application commands      thread-safe methods on runtime, controller, and callbacks
async studio worker       wrap RobotRuntime in a worker, or keep async behind Robot adapter
multi-system autonomy     defined separately in composite_robot_architecture.md
```

This design keeps the production robot loop small and predictable. It leaves room for richer controllers and remote inference without turning callbacks into a hidden control architecture and without forcing the loop itself to become async.
