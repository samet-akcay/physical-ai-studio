# PhysicalAI Robot Stack: Architecture Vision

This document describes how PhysicalAI structures the layers between a learned policy and a physical robot, and how those layers fit under a future multi-system robot.

It is a vision document. It is intentionally short, contains no APIs and no pseudocode, and is meant to remain useful as the implementation evolves.

---

## 1. Why This Document Exists

PhysicalAI today supports running learned policies on robots. As the system grows toward more capable robots — manipulators with vision-language conditioning, mobile bases, eventually humanoids — the question is not "how do we run a policy" but "how do we organize the stack so that policies, planners, perception, safety, teleop, and Studio workflows can all coexist without one component swallowing the others."

This document fixes the layer boundaries, the dependency direction, and the names. It does not specify APIs.

---

## 2. Guiding Principles

1. **VLA Policy inference is one component, not the whole robot.** The runtime that drives a robot from a learned policy is a critical layer, but it is not the top of the stack and it is not the only way to drive a robot.
2. **Each layer has one owner.** Policy math, denoising correction, scheduling, action dispatch, and product workflows are separate concerns. Each lives in exactly one layer.
3. **Dependencies point downward.** Higher layers depend on lower layers. Lower layers never import higher layers.
4. **Hardware boundaries are stable.** `Robot` and `Camera` are the hardware seams. Everything above them can change; they should not.
5. **Product workflows are not infrastructure.** Recording, teleop, HIL, DAgger, and sentry are Studio-level workflows. They configure the runtime; they do not live inside it.
6. **Reserve names for layers that don't exist yet.** Today's runtime should be narrow enough that a future top-level orchestrator can be added above it without renaming or restructuring.

---

## 3. The Layered Picture

```text
+-------------------------------------------------------------------+
|                          RobotSystem                              |
|   (future) coordinates planner, perception, memory, safety,       |
|   locomotion, manipulation, and one or more PolicyRuntimes        |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
|                       Studio Strategies                           |
|   recording, teleop, HIL, DAgger, sentry, highlight               |
|   configure runtime extension points; do not own runtime          |
+-------------------------------------------------------------------+
                                 |
                                 v
+-------------------------------------------------------------------+
|                         PolicyRuntime                             |
|   per-tick control loop for a learned policy on a robot:          |
|     - timing (FPS)                                                |
|     - observation assembly (robot state + camera frames)          |
|     - action queue + arbitration + safety gate                    |
|     - dispatch to robot                                           |
|     - episode lifecycle, callbacks, shutdown                      |
+-------------------------------------------------------------------+
       |                |                  |                |
       v                v                  v                v
+------------+  +---------------+  +---------------+  +-------------+
| Execution  |  | InferenceModel|  |     Robot     |  |   Camera    |
| sync/async |  | policy API    |  | hardware ABC  |  | capture ABC |
| scheduling |  | + Runners     |  |               |  |             |
|            |  | + Guidance    |  |               |  |             |
+------------+  +---------------+  +---------------+  +-------------+
```

The dotted line above `PolicyRuntime` is a future expansion. The solid lines are what PhysicalAI builds and maintains.

---

## 4. What Each Layer Owns

| Layer | Owns | Does Not Own |
|---|---|---|
| `RobotSystem` *(future)* | high-level task reasoning, perception orchestration, world state, safety supervision, coordination across multiple subsystems | per-tick policy execution, hardware I/O details |
| Studio strategies | product workflows: recording, teleop arbitration, intervention state machines, sentry policies | timing, dispatch, policy math |
| `PolicyRuntime` | per-tick loop, observation assembly, action queue, arbitration, safety gate, dispatch, episode/lifecycle, callbacks | policy math, denoising, dataset writing details, intervention semantics |
| `Execution` | when to ask the policy for actions; sync vs async vs remote | how the policy computes actions; how actions are dispatched |
| `InferenceModel` + `Runner` + `Guidance` | how the policy computes actions and chunks; denoising correction | timing, robot I/O, recording |
| `Robot` | observe and command one robot's hardware | cameras, policies, runtimes |
| `Camera` | connect, read frames | robot state, observations |

The single most important rule: **`PolicyRuntime` does not know what DAgger means, what a VLM is, or what a dataset row looks like.** It exposes seams; higher layers use them.

---

## 5. Naming and Reserved Names

These names are fixed by this document. They appear across every other doc, every API, and every diagram.

| Name | Status | Meaning |
|---|---|---|
| `Robot` | exists | hardware boundary for a robot |
| `Camera` | exists | hardware boundary for a camera |
| `InferenceModel` | exists | user-facing policy API |
| `InferenceRunner` | exists | policy computation strategy |
| `Guidance` | planned | denoising-time correction (e.g. RTC) |
| `Execution` | planned | scheduling policy (sync, async, remote) |
| `PolicyRuntime` | planned | per-tick control loop for a policy on a robot |
| Studio strategies | partially exists | product workflows above `PolicyRuntime` |
| `RobotSystem` | reserved | future top-level orchestrator across multiple subsystems |

Names deliberately not used and why:

- **`ActionRuntime`** — under-sells observation and timing ownership; reads as "executes actions handed to it."
- **`ControlLoop`** — too generic; locomotion, whole-body control, and servo loops are also control loops.
- **`PolicyController`** — collides with classical control (PID, MPC, WBC); misleading in a robot stack.
- **`RolloutRuntime`** — implies eval/data collection; the runtime is the production execution layer.
- **`RobotRuntime`** — sounds like the top-level runtime for the whole robot; reserved space for `RobotSystem` instead.

---

## 6. Dependency Direction

```text
RobotSystem      ->  Studio strategies, PolicyRuntime, InferenceModel
Studio           ->  PolicyRuntime, InferenceModel
PolicyRuntime    ->  InferenceModel, Execution, Robot, Camera
Execution        ->  InferenceModel
InferenceModel   ->  InferenceRunner, Guidance
Robot            ->  (nothing above it)
Camera           ->  (nothing above it)
```

Concretely:

- `Robot` and `Camera` never import from `physicalai.runtime`, `physicalai.inference`, or `physicalai.studio`.
- `physicalai.inference` never imports from `physicalai.runtime` or `physicalai.studio`.
- `physicalai.runtime` never imports from `physicalai.studio`.
- A future `physicalai.system` (or wherever `RobotSystem` lives) may depend on all of the above.

This direction is what allows each layer to be tested, replaced, or reused independently.

---

## 7. What `RobotSystem` Is For (Future)

`RobotSystem` is the layer where a complex robot's subsystems are coordinated. It is not part of the first implementation. It exists in this document so that the layers below it stay narrow enough to fit under it cleanly when it is built.

A `RobotSystem` would be responsible for:

- accepting a high-level task or goal
- coordinating a planner (e.g. a VLM) with one or more policies
- managing perception services and shared world state
- supervising safety across subsystems
- routing user intent (teleop, intervention, pause) to the right subsystem
- starting, stopping, and supervising one or more `PolicyRuntime` instances

`RobotSystem` is **not** a single FPS loop. It is a coordinator. The fast loops live in `PolicyRuntime` (for learned manipulation) and in future runtimes for locomotion, whole-body control, or servo-rate control.

This document does not specify `RobotSystem`'s API. Its scope, name, and place in the hierarchy are reserved here so that the design of `PolicyRuntime` can stop at the right boundary.

---

## 8. Current State vs Target State

### Current state (today, in `application/`)

```text
WebSocket
   |
   v
RobotControlWorker  (thread, owns FPS loop, state, events, queue)
   |
   +-- EnvironmentIntegration  (robot + cameras + observation formatting)
   +-- SyncMixedModelIntegration  (InferencePoller + QueueMixer)
   |       |
   |       v
   |   ModelWorker  (separate process)
   |       |
   |       v
   |   InferenceModel.select_action(...)
   |
   +-- RecordingMutation  (dataset writing)
```

The current implementation works and is in production. It bundles concerns that the target architecture separates: timing, robot I/O, async inference, action smoothing, recording, teleop arbitration, and WebSocket transport all live in or directly under `RobotControlWorker`.

### Target state

```text
WebSocket transport (thin)
   |
   v
PolicyRuntime
   |
   +-- Robot
   +-- Camera(s)
   +-- Execution (sync | async | remote-via-process-pool)
   |       |
   |       v
   |   InferenceModel
   |       |
   |       +-- InferenceRunner (SinglePass | FlowMatching | TemporalEnsemble)
   |       +-- Guidance (RTC, when applicable)
   |
   +-- ActionArbiter      (policy | teleop | hold | emergency)
   +-- ActionFilter(s)    (safety gate)
   +-- Callback(s)        (recording, telemetry, reporting)
```

Each box in the target diagram has one owner. The existing functionality of `RobotControlWorker` is preserved, but distributed across these seams: teleop becomes an `ActionArbiter` mode, recording becomes a `Callback`, async inference becomes an `Execution`, and the WebSocket transport becomes a thin shell that translates client events into `PolicyRuntime` calls.

The design document specifies how this distribution is done and how the migration from current to target proceeds.

---

## 9. Non-Goals of This Vision

This document deliberately does not:

- specify any API
- choose between threading and multiprocessing for `Execution`
- describe RTC math
- pick a manifest schema
- define `ActionChunk` shape
- describe Studio strategy implementations
- propose a phased rollout

Those belong in the design document and in future implementation docs. Putting them here would cause this document to rot the first time an API changes.

---

## 10. How To Use This Document

- **New contributors** read this first to understand where things live and why.
- **Reviewers** of any feature touching robot control should check that the feature lands in the right layer per §4 and respects §6.
- **Designers** of new features cite sections of this document when justifying placement.
- **Renaming** any of the names in §5 requires updating this document first.

---

## 11. Companion Documents

- *(future)* migration plan for `application/` once the design is reviewed and the API shapes are stable.
- *(future)* `RobotSystem` design, when there is concrete pressure for a multi-subsystem orchestrator.
