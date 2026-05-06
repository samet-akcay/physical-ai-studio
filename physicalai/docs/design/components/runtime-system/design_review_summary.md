# Policy Runtime Design — 1-Page Summary

**Full design.** [`policy_runtime_design.md`](./policy_runtime_design.md), [`policy_server_design.md`](./policy_server_design.md).
**Gap analysis.** [`inference_comparison_report.md`](./inference_comparison_report.md).

## What we're proposing

A new `physicalai.runtime` package owning the per-tick control loop. `InferenceModel` keeps its current API plus `predict_action_chunk()` and `close()`. The runtime is composable from three pieces: `InferenceModel` (policy math), `Execution` (transport — sync, async-thread, async-process, remote), and `PolicyRuntime` (loop, queue, callbacks, timing). Each piece is independently testable.

Studio's `application/backend/` becomes a thin shell that constructs a `PolicyRuntime`, attaches callbacks for recording/telemetry, and forwards user events. Per-tick logic moves out of `RobotControlWorker`.

## What we're adding

| | What | Where |
|---|---|---|
| Loop | `PolicyRuntime` with `run()` / `stop()` / `from_config()` | `physicalai.runtime` |
| Transport | `Execution` protocol + `Sync` / `Async(thread)` / `Async(process)` / `Remote` | `physicalai.runtime.execution` |
| Chunks | `ActionQueue` + `ChunkSmoother` + `RTCQueueMerger` | `physicalai.runtime.action_queue` |
| RTC | `Guidance` protocol + `RTC()` composed into `FlowMatching` | `physicalai.inference.guidance` |
| Hooks | `Callback` base + documented hook points | `physicalai.runtime.callbacks` |
| CLI | `physicalai run` subcommand on a new runtime-side `physicalai` console script (jsonargparse, no Lightning) | `physicalai/src/physicalai/cli/` |
| Remote | `RemoteExecution` + `PolicyServer` (gRPC streaming) | `physicalai.runtime.execution.remote` |

## What we're explicitly not adding

- `ObservationAssembler`, `ActionArbiter`, `ActionFilter`, `ActionInterpolator`, `ShutdownPolicy` — deferred until two consumers need them. Today's needs are met by callbacks and the `return_to_home: bool` kwarg.
- Strategy classes (sentry / dagger / highlight / HIL) — these are product workflows. They live in the consumer, composed from `PolicyRuntime` + callbacks. Composition sketches in `policy_runtime_design.md` §19.
- `torch.compile`, `TwoPhase` runner, server-side smoothing, multi-tenant single-server.

## Why this shape

1. **`Execution` is one concept, four transports.** Sync, thread, process, remote share one interface. `PolicyRuntime` does not learn about transport. Same code path supports today's `ModelWorker` (process) and tomorrow's remote GPU.
2. **RTC composes, doesn't subclass.** `FlowMatching(guidance=RTC()) + AsyncExecution() + ActionQueue(merger=RTCQueueMerger())`. Each piece works without the others. LeRobot's `RTCInferenceEngine` lumps them together; we don't.
3. **Strategies live in the consumer.** The loop is workflow-agnostic. Adding a new product workflow is a callback + a small wrapper in the consumer, not a runner subclass.

## Phases

| | Scope | Acceptance |
|---|---|---|
| 0 | Correctness fixes (image preprocessing parity, processor state) | Known-divergent checkpoint loads correctly or fails loudly |
| 1 | `InferenceModel` additions (`predict_action_chunk`, `close`), `Guidance`, `FlowMatching`, `TemporalEnsemble` | `predict_action_chunk` returns documented dict; `RTC` with no overlap is identity |
| 2 | `physicalai.runtime` core + `SyncExecution` + `AsyncExecution(thread)` + `physicalai run` CLI | Standalone example runs end-to-end at target FPS |
| 3 | `AsyncExecution(process)` + RTC end-to-end + `RTCQueueMerger` | Flow policy runs with RTC without chunk-boundary discontinuities |
| 4 | Hardening (numerical equivalence tests, latency tracking accuracy) | Smoothing matches reference; latency estimate within 1 frame |
| 5 | `RemoteExecution` + `PolicyServer` + `physicalai serve` CLI | Remote run produces same actions as `AsyncExecution(process)` |
| 6 | Deferred components, when consumer demand justifies them | Per-component graduation conditions in §16 |

## Open questions for the team

1. `TemporalEnsemble` output: one smoothed action per tick, or smoothed chunk?
2. `AsyncExecution` cancellation: keep no-cancel default or add `cancel_inflight=True` opt-in?
3. Process-transport reader: dedicated reader thread, or poll inside `maybe_request`?
4. `ObservationAssembler` protocol: needed in Phase 2, or callbacks suffice indefinitely?
5. Remote transport: gRPC streaming (recommended) vs HTTP/2+JSON; multi-tenancy timing; auth posture.
