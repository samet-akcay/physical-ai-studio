# Composite Robot Architecture

This document defines the **multi-system autonomy architecture** for composite robots — humanoids, mobile manipulators, and any robot that runs more than one decision-making subsystem concurrently (VLA + locomotion + perception + world model + planner + …).

It builds on the production single-rate runtime defined in [`robot_runtime_architecture.md`](./robot_runtime_architecture.md) (Doc A). It does not replace it. Composite autonomy is **one `Controller` implementation** — `AutonomyController` — that runs inside `RobotRuntime` and returns one `RobotAction` per tick. Everything below describes what lives inside that controller.

This is intentionally future-facing. No code in this document should be implemented before a concrete composite robot integration (e.g., a Unitree G1) is in scope. The purpose of writing it now is to ensure Doc A's contracts (`Controller`, `Robot`, `RobotAction`, `Observation`) do not paint us into a corner when composite autonomy lands.

---

## 1. Scope And Non-Goals

### Scope

A reusable composition layer for robots whose autonomy combines several systems with different rates, latencies, and responsibilities, such as:

- A **VLA policy** producing arm/hand targets at ~10–30 Hz
- A **locomotion stack** producing base twists or gait goals at ~100–500 Hz
- **Perception** producing scene state at ~10–30 Hz
- A **world model** maintaining occupancy / object memory at ~5–10 Hz
- A **task planner** selecting goals at ~0.1–1 Hz

The layer must:

1. Run each subsystem at its own rate without blocking the control tick.
2. Share state between subsystems through a **typed, timestamped blackboard**, not direct calls.
3. Produce one coherent `RobotAction` per `RobotRuntime` tick by **arbitrating effector-scoped commands** from action-producing subsystems.
4. Degrade safely when a subsystem is slow, unhealthy, or absent.
5. Stay framework-light: no mandatory ROS 2, no mandatory behavior-tree dependency, no GXF runtime.

### Non-Goals

- Replacing low-level control (joint servoing, gait MPC, balance). Those live in the `Robot` driver or the vendor stack.
- Replacing `RobotRuntime`. The outer loop, lifecycle, callbacks, safety, and dispatch stay in Doc A.
- Defining the policy/inference plumbing. That is `PolicyController` + `InferenceExecution` from Doc A §5.
- Becoming a general distributed actor framework. Multi-process / multi-host orchestration is a possible future extension; the initial layer is single-process.
- Replacing ROS 2 where ROS 2 already exists. ROS 2 is an integration target, not a competitor.
- Defining a typed `RobotAction` class hierarchy. Use the namespaced mapping form from [`../robot-interface.md`](../robot-interface.md#action-evolution-from-npndarray-to-namespaced-mappings).

---

## 2. Why Separate From Doc A

Three reasons composite autonomy does **not** belong inside `robot_runtime_architecture.md`:

1. **Different cadence.** Doc A is one rate; composite is many.
2. **Different contracts.** Doc A's `Controller` is `Observation -> RobotAction`. Composite needs subsystem lifecycle, blackboard semantics, freshness, and arbitration — none of which the policy/teleop/HIL/DAgger workflows need.
3. **Different shipping order.** Doc A ships now and powers the existing Studio worker. Doc B ships when there is a real composite robot to drive. Mixing them would force composite concepts into every reader of the production design.

The boundary between the two docs:

```text
Doc A:
  RobotRuntime  ->  Controller  ->  RobotAction  ->  Robot

Doc B:
  AutonomyController implements Controller
  AutonomyController owns:
    subsystem scheduler
    blackboard
    arbitration
    perception / world / planner / VLA / locomotion subsystems
```

`RobotRuntime` does not change.

---

## 3. Prior-Art Survey

A comparison of systems we considered borrowing from. Each row identifies what to **steal** and what to **avoid**.

| System                              | Steal                                                                                           | Avoid                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **ROS 2 / rclpy**                   | lifecycle states, pub/sub mental model, QoS / deadline / liveliness thinking, actions/services for long-running goals, TF-style frame naming, timestamped messages | mandatory `rclpy` dependency, leaking ROS message types into PhysicalAI core, executor complexity for simple robots |
| **NVIDIA Isaac GXF**                | explicit graph of components, scheduler abstraction, timestamped tensors                        | NVIDIA-specific heavy runtime, tight coupling to Isaac SDK                  |
| **MuJoCo MJX dataflow**             | clean state/action arrays, sim batchability, normalization discipline                           | using a sim dataflow as a real-robot runtime                                |
| **Behavior trees (py_trees, BehaviorTree.CPP)** | tick/status model, blackboard pattern, task-level arbitration                       | forcing every subsystem to be a BT node                                     |
| **LeRobot**                         | simple robot/policy/dataset ergonomics, naming                                                  | using it as an autonomy architecture (it isn't one)                         |
| **Boston Dynamics SDK patterns**    | leases (single writer), e-stop, command feedback, time sync, command authority                  | proprietary robot-specific assumptions                                      |
| **Open Robotics `ros2_control`**    | hardware abstraction vs controller manager separation, controller switching                     | replacing robot drivers or low-level controllers inside PhysicalAI          |

### ROS 2 Verdict

**Use ROS 2 as an optional integration boundary, not the core runtime.**

Why not mandatory ROS 2:

- Simple robots (SO-101) should not need a ROS install.
- `rclpy` executors add complexity that hurts research notebooks/scripts.
- PhysicalAI must remain robot-agnostic.

How ROS 2 enters the picture for complex robots: as a **subsystem adapter**.

```text
ROS2LocomotionSubsystem
  subscribes to robot state topics
  publishes velocity / gait commands
  writes health + state to the Blackboard
  implements RuntimeSubsystem
```

The autonomy graph remains PhysicalAI-native; ROS 2 lives at one boundary.

---

## 4. AutonomyController Inside RobotRuntime

`AutonomyController` is the only thing `RobotRuntime` sees. It satisfies Doc A's `Controller` protocol:

```python
class AutonomyController:
    def start(self) -> None: ...
    def update(self, observation: Observation) -> RobotAction: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
```

Inside, it owns:

```text
AutonomyController
  scheduler         drives subsystems at their own rates
  blackboard        typed, timestamped cache of subsystem outputs
  subsystems
    perception      InfoSubsystem  (scene, detections)
    world_model     InfoSubsystem  (occupancy, object memory)
    planner         InfoSubsystem  (intent, goals)
    vla             ActionSubsystem (arm/hand RobotAction fragments)
    locomotion      ActionSubsystem (base RobotAction fragments)
  arbiter           merges action fragments into one RobotAction
```

`update(observation)` is fast and non-blocking: it publishes the observation, ticks any subsystems whose schedule fires this loop iteration, reads the latest action fragments from the blackboard, runs the arbiter, and returns one `RobotAction`. Slow subsystems do not stall the tick — the arbiter uses the most recent fresh fragment per effector, or applies the configured degrade behavior.

```text
RobotRuntime tick:
  observation = robot.get_observation() + cameras + runtime fields
  controller.update(observation):
    blackboard.publish("observation", observation)
    scheduler.tick(now)                 # may run 0..N subsystems this iteration
    fragments = blackboard.read_action_fragments()
    return arbiter.merge(fragments, observation, now)
  robot.send_action(action)
```

---

## 5. RuntimeSubsystem Protocol

Two flavors. Both share lifecycle and health.

### 5.1 Common lifecycle

```python
class RuntimeSubsystem(Protocol):
    name: str
    rate_hz: float | None      # None = event-driven, no fixed schedule

    def start(self, ctx: SubsystemContext) -> None: ...
    def stop(self) -> None: ...
    def reset(self) -> None: ...
    def health(self) -> SubsystemHealth: ...
```

`SubsystemContext` exposes the blackboard (read/write), a clock, and configuration. `SubsystemHealth` carries a status enum (`OK | DEGRADED | FAILED | STARTING | STOPPED`), a timestamp, and a free-form message.

### 5.2 InfoSubsystem (perception, world model, planner)

Produces intermediate state into the blackboard. Does **not** produce `RobotAction`.

```python
class InfoSubsystem(RuntimeSubsystem, Protocol):
    def step(self, ctx: SubsystemContext) -> None:
        """Read inputs from blackboard, write outputs to blackboard."""
```

Examples:

```python
class PerceptionSubsystem(InfoSubsystem):
    name = "perception"
    rate_hz = 30.0
    def step(self, ctx):
        obs = ctx.blackboard.read("observation")
        scene = self._detect(obs.images)
        ctx.blackboard.write("scene", scene, ttl_s=0.2)
```

### 5.3 ActionSubsystem (VLA, locomotion, scripted)

Produces a `RobotAction` fragment for one or more effectors.

```python
class ActionSubsystem(RuntimeSubsystem, Protocol):
    effectors: frozenset[str]   # e.g., {"left_arm", "right_arm"}
    priority: int               # higher wins on conflict (see Arbiter)

    def step(self, ctx: SubsystemContext) -> ActionFragment:
        """Return action fragment for this subsystem's effectors."""
```

```python
@dataclass(frozen=True)
class ActionFragment:
    source: str                       # subsystem name
    fragment: Mapping[str, Any]       # effector-scoped, see robot-interface.md
    timestamp: float                  # producer time
    valid_until: float | None = None  # explicit freshness deadline
    confidence: float = 1.0
```

Example:

```python
class VLAArmSubsystem(ActionSubsystem):
    name = "vla_arms"
    rate_hz = 20.0
    effectors = frozenset({"left_arm", "right_arm"})
    priority = 10

    def step(self, ctx):
        obs   = ctx.blackboard.read("observation")
        scene = ctx.blackboard.read("scene", default=None)
        intent = ctx.blackboard.read("intent", default=None)
        enriched = enrich(obs, scene=scene, intent=intent)
        # PolicyController-style internals omitted for brevity
        chunk = self._policy.predict_action_chunk(enriched)
        a = chunk[0]
        return ActionFragment(
            source=self.name,
            fragment={
                "left_arm":  {"joint_positions": a[:7], "mode": "position"},
                "right_arm": {"joint_positions": a[7:14], "mode": "position"},
            },
            timestamp=ctx.clock.now(),
            valid_until=ctx.clock.now() + 0.15,
        )
```

A VLA `ActionSubsystem` may internally reuse Doc A's `PolicyController`, `InferenceExecution`, and `ActionQueue`. Composite autonomy does not reinvent the inference plumbing.

---

## 6. Blackboard

A typed, timestamped, in-process key-value cache with freshness semantics.

```python
class Blackboard:
    def write(self, key: str, value: Any, *, ttl_s: float | None = None) -> None: ...
    def read(self, key: str, *, default: Any = _MISSING, max_age_s: float | None = None) -> Any: ...
    def read_entry(self, key: str) -> BlackboardEntry | None: ...
    def keys(self) -> Iterable[str]: ...
    def subscribe(self, key: str, callback: Callable[[BlackboardEntry], None]) -> Subscription: ...
```

```python
@dataclass(frozen=True)
class BlackboardEntry:
    key: str
    value: Any
    timestamp: float
    ttl_s: float | None
    writer: str
```

Rules:

- Writes are atomic per key; readers always see a consistent snapshot of one entry.
- A read with `max_age_s` returns the default when the entry is older than the threshold (data is stale, treat as absent).
- Single writer per key by convention. Multi-writer keys require an explicit merge function registered at construction.
- Subscriptions fire synchronously on the writer's thread. Subscribers must not block.
- The blackboard is **not** a message queue. Consumers see only the latest value, never a history.

This is deliberately closer to a behavior-tree blackboard than a ROS topic. History/replay belongs in the recording callback (Doc A §6.3) or a separate logger, not the blackboard.

---

## 7. Multi-Rate Scheduler

A cooperative scheduler that runs subsystems at their declared rates from a single thread by default, with per-subsystem worker-thread escape hatches.

### Default: cooperative

```text
scheduler.tick(now):
  for subsystem in subsystems:
    if now >= subsystem.next_tick:
      try:
        subsystem.step(ctx)
        subsystem.next_tick = now + 1.0 / subsystem.rate_hz
      except Exception as e:
        record_failure(subsystem, e)
        # do not raise; arbiter handles degrade
```

The cooperative scheduler is enough when every subsystem's `step()` is faster than the smallest period it shares with `RobotRuntime`'s tick. This is realistic for perception/planner subsystems on modern hosts.

### Worker-thread subsystems

Slow subsystems (heavy VLA inference, world-model updates) declare `execution="thread"` (or `"process"`). The scheduler then:

- Owns one background worker per such subsystem.
- The cooperative `tick()` only **enqueues** a step request; the worker runs `step()` and writes results to the blackboard.
- The control tick continues using the latest blackboard entry, never blocking on the worker.

This mirrors Doc A's `InferenceExecution` design and is the same idea: the slow thing happens off the control thread, the consumer reads cached output.

### Real-time considerations

True real-time scheduling (preempt, deadline) is out of scope for the Python-side scheduler. Time-critical subsystems (locomotion balance, joint servoing) belong in the `Robot` driver or vendor stack, exposed to the autonomy layer through a `RuntimeSubsystem` adapter that publishes status.

### Failure handling

Per-subsystem failure modes:

| Failure                                | Scheduler behavior                                                |
| -------------------------------------- | ----------------------------------------------------------------- |
| `step()` raises                        | log, mark `health = FAILED`, retry next tick                      |
| `step()` exceeds period repeatedly     | mark `health = DEGRADED`, keep running                            |
| Worker thread/process dies             | mark `health = FAILED`, `AutonomyController.health()` reflects it |
| Subsystem reports `health = FAILED`    | scheduler keeps ticking; arbiter applies degrade for that effector |

The scheduler **never** kills the control loop. Loop termination is `RobotRuntime`'s job (Doc A §8.5).

---

## 8. ActionArbiter

Merges effector-scoped action fragments into one `RobotAction`. Resolves conflicts when multiple subsystems write to the same effector.

```python
class ActionArbiter(Protocol):
    def merge(
        self,
        fragments: Sequence[ActionFragment],
        observation: Observation,
        now: float,
    ) -> RobotAction: ...
```

### Default arbiter rules

1. **Filter stale fragments.** Drop any fragment with `valid_until < now`.
2. **Group by effector.** Each fragment contributes to one or more effector keys.
3. **Resolve conflicts.** For an effector claimed by multiple fragments, the **higher `priority`** wins. Ties broken by most recent `timestamp`.
4. **Per-effector degrade.** If no fresh fragment is available for an effector that the robot expects, apply the configured degrade policy:
   - `hold` — repeat last sent value for that effector.
   - `safe` — send a safe default (e.g., zero base twist, hands open).
   - `omit` — send no command for that effector this tick (the driver decides).
5. **Compose** the surviving per-effector commands into one mapping and return it.

### Authority / leasing

For effectors that must have a single writer at a time (typical for base motion to avoid fighting subsystems), the arbiter supports **leases**:

```python
arbiter.grant_lease(effector="base", subsystem="locomotion")
```

While a lease is active, fragments from other subsystems for that effector are rejected with a logged warning. Leases are the BD-SDK pattern adapted to Python-process scope and are the simplest way to prevent VLA / planner / locomotion fights.

### Safety boundary

The arbiter is **not** the safety layer. Doc A's `SafetyLayer` still runs after `callbacks.before_send_action` and is the last gate before `robot.send_action`. The arbiter resolves *intent conflicts*; safety enforces *hard constraints*.

---

## 9. Effector-Scoped Actions

Composite robots use the namespaced `RobotAction` form from [`../robot-interface.md`](../robot-interface.md#namespaced-action-examples):

```python
{
    "base":       {"twist": np.array([vx, vy, wz])},
    "torso":      {"joint_positions": q_torso, "mode": "position"},
    "left_arm":   {"joint_positions": q_la,    "mode": "position"},
    "right_arm":  {"joint_velocities": qd_ra,  "mode": "velocity"},
    "left_hand":  {"joint_positions": q_lh},
    "right_hand": {"grasp_force": 0.6},
    "head":       {"joint_positions": q_head},
}
```

Subsystems own subsets of these keys via `effectors`. The composite `Robot` driver routes each effector to the right hardware controller. Effectors absent from the action mean "no command this tick"; it is the driver's job to decide whether to hold, decay, or refuse.

---

## 10. Worked Example: Unitree G1-Style Humanoid

```python
runtime = RobotRuntime(
    robot=G1Robot(...),                       # composite Robot driver
    controller=AutonomyController(
        subsystems=[
            PerceptionSubsystem(rate_hz=30),
            WorldModelSubsystem(rate_hz=10, execution="thread"),
            PlannerSubsystem(rate_hz=1),
            VLAArmSubsystem(                  # left+right arms, hands
                rate_hz=20,
                policy_controller=PolicyController(
                    model=arm_model,
                    execution=AsyncInferenceExecution(transport="process"),
                ),
                effectors={"left_arm", "right_arm", "left_hand", "right_hand"},
                priority=10,
            ),
            ROS2LocomotionSubsystem(          # base + balance via vendor ROS stack
                effectors={"base", "torso"},
                priority=20,
            ),
            HeadGazeSubsystem(rate_hz=10, effectors={"head"}, priority=5),
        ],
        arbiter=DefaultActionArbiter(
            degrade={
                "base":  "safe",   # zero twist if locomotion is stale
                "torso": "hold",
                "left_arm":   "hold",
                "right_arm":  "hold",
                "left_hand":  "hold",
                "right_hand": "hold",
                "head":  "omit",
            },
            leases={"base": "locomotion"},
        ),
    ),
    fps=50,
    safety=G1SafetyLayer(),
    callbacks=[RecordingCallback(...), TelemetryCallback(...)],
)
runtime.run()
```

What happens per tick (50 Hz):

- `RobotRuntime` reads `Robot.get_observation()` (G1 driver returns base state, joints, IMU, cameras).
- `AutonomyController.update()` publishes the observation.
- The scheduler ticks subsystems whose deadline has fired:
  - Perception runs every ~33 ms in-line.
  - World model runs in a worker thread; control tick reads its latest output.
  - Planner runs every ~1 s in-line.
  - VLA runs every ~50 ms in a worker process; arm/hand fragments updated.
  - Locomotion is event-driven from a ROS subscription; base/torso fragments updated each ROS callback.
  - Head subsystem runs every ~100 ms in-line.
- The arbiter merges fragments using the lease (locomotion owns `base`), per-effector priorities, and freshness; absent fragments fall back to their `degrade` policy.
- `RobotRuntime` runs callbacks → safety → `G1Robot.send_action(merged_action)`.

`RobotRuntime` does not change. Studio's `RobotControlWorker` does not change. The composite-robot complexity is fully contained in `AutonomyController` and the `G1Robot` driver.

---

## 11. AutonomyRuntime: Why Not (Yet)

A separate `AutonomyRuntime` is **not** introduced. Reasons to keep composite autonomy as a `Controller`:

- One outer loop is easier to reason about, record, and supervise.
- Doc A's lifecycle, safety, callbacks, and error handling apply unchanged.
- `RobotRuntime` already supports thread-safe controller swap (Doc A §9), enabling autonomy hot-load.

A future `AutonomyRuntime` becomes warranted only when one of the following is real:

- Multi-process subsystem lifecycle that PhysicalAI must own end-to-end.
- A native ROS 2 graph that owns the control loop and PhysicalAI is a node inside it.
- Hard real-time scheduling outside the Python loop (e.g., a separate C++ control process with PhysicalAI as a planner client).

When that happens, `AutonomyController` becomes the in-process façade to a separate runtime; the contract upward (`Controller`) does not have to change.

---

## 12. Manifest And Configuration

Composite manifests extend the per-robot schema with per-effector entries. A sketch:

```json
{
  "format": "policy_package",
  "version": "1.0",
  "robots": [
    {
      "name": "g1",
      "type": "Unitree-G1",
      "effectors": {
        "base":       {"command": "twist",            "shape": [3]},
        "torso":      {"command": "joint_positions",  "shape": [4]},
        "left_arm":   {"command": "joint_positions",  "shape": [7], "mode": "position"},
        "right_arm":  {"command": "joint_positions",  "shape": [7], "mode": "position"},
        "left_hand":  {"command": "joint_positions",  "shape": [6]},
        "right_hand": {"command": "joint_positions",  "shape": [6]},
        "head":       {"command": "joint_positions",  "shape": [2]}
      }
    }
  ],
  "cameras": [...]
}
```

A composite policy manifest would declare which **effector subsets** it produces (e.g., a bimanual VLA that produces `left_arm` + `right_arm` + hands). The `AutonomyController` config wires each subsystem to its effector subset and registers the corresponding lease/priority.

The exact composite manifest schema is deferred until the first composite driver is integrated. Until then, the existing flat-vector manifest (Doc A) remains canonical.

---

## 13. What This Doc Does NOT Define

Deferred until a concrete need arises:

- `AutonomyRuntime` (separate runtime; see §11).
- Typed `RobotAction` / `Effector` dataclass hierarchy (use mappings; see [`../robot-interface.md`](../robot-interface.md)).
- Cross-process / cross-host blackboard (start in-process).
- Behavior-tree dependency for arbitration (the arbiter is a function, not a BT).
- ROS 2 message-type imports anywhere outside ROS 2 subsystem adapters.
- Generic robot transport / data-plane abstraction (see Doc A §10).
- Real-time scheduling guarantees from the Python-side scheduler.
- Composite manifest schema beyond the §12 sketch.

---

## 14. Decision Summary

```text
composite autonomy contract        AutonomyController implements Doc A's Controller
no new outer runtime               RobotRuntime stays the only outer loop
subsystems share state             Blackboard (typed, timestamped, single-writer-by-default)
scheduling                         cooperative by default, worker thread/process per subsystem on demand
multi-source action assembly       ActionArbiter with priority + freshness + leases
effector contract                  namespaced RobotAction mappings (../robot-interface.md)
ROS 2                              optional integration via subsystem adapters; not foundational
real-time control                  in the Robot driver / vendor stack, exposed via subsystems
relationship to PolicyController   ActionSubsystems may internally reuse PolicyController + InferenceExecution
implementation trigger             defer until a concrete composite-robot integration is in scope
```

The composite layer adds zero burden to single-arm robots, keeps Doc A's small surface intact, and provides a clear path for humanoids and mobile manipulators when the first one lands.
