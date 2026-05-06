# PolicyServer and RemoteExecution: Remote Inference Design

**Status.** Design fixed; build target Phase 5 of the runtime build sequence.
**Companion to.** `policy_runtime_design.md` (Phase 5 acceptance criteria live there in §15).
**Audience.** Engineers building or reviewing PhysicalAI's remote inference path.

---

## 1. Purpose

This document specifies how PhysicalAI runs the policy on a host separate from the robot. It covers two components that ship together:

- **`RemoteExecution`** — the client-side `Execution` implementation used by `PolicyRuntime` on the robot host.
- **`PolicyServer`** — the server-side process that loads the `InferenceModel` and serves inference requests over the network.

The design is not part of the initial Phase 0 / Phase 1 runtime build. It is fixed now because the requirement is committed and the `Execution` interface position needs to be reserved before downstream consumers commit to async-process-only deployments. The `Execution` contract that `RemoteExecution` satisfies is defined in `policy_runtime_design.md` §10; this document does not redefine it.

The design adopts LeRobot's `PolicyServer` / `RobotClient` split, restructured to fit PhysicalAI's `Execution` / `PolicyRuntime` boundaries. Differences from LeRobot are intentional and called out in §9.

---

## 2. Process Model

Two processes, possibly on different hosts:

- **Robot host.** Runs `PolicyRuntime` with a `RemoteExecution` instance in place of `SyncExecution` or `AsyncExecution`. Owns `Robot`, cameras, and the `ActionQueue`. Does not load the policy.
- **Server host.** Runs `PolicyServer`. Owns the `InferenceModel`, runs warmup, and serves `predict_action_chunk` / `select_action` over the network. Does not own a `Robot` or an `ActionQueue`.

The split mirrors `AsyncExecution(transport="process")`: the robot side owns the loop and the queue; the inference side owns the model. The only differences are the transport (network instead of `mp.Queue`) and the failure modes that come with it.

```
┌───────────── Robot host ─────────────┐         ┌──────── Server host ─────────┐
│ PolicyRuntime                        │         │ PolicyServer                 │
│  ├─ Robot, cameras                   │         │  ├─ InferenceModel           │
│  ├─ ActionQueue                      │         │  ├─ runner (FlowMatching,    │
│  └─ RemoteExecution ──── network ────┼─────────┼──────  TemporalEnsemble, …)  │
│       (client)                       │   gRPC  │  └─ Guidance (RTC, …)        │
└──────────────────────────────────────┘         └──────────────────────────────┘
```

---

## 3. Robot-Host Responsibilities (`RemoteExecution`)

`RemoteExecution` is an `Execution` implementation. It satisfies the same contract as `SyncExecution` and `AsyncExecution`:

- `start(action_queue, model)` opens the connection and runs the version handshake. The `model` argument is a **stub** (`RemoteInferenceModel`) that holds the connection and forwards calls; it is not the real `InferenceModel`.
- `maybe_request(observation)` serializes the observation snapshot and dispatches an RPC. Single-flight semantics (no in-flight request → send; in-flight → skip) are unchanged from `AsyncExecution`.
- A small reader thread (or async task) drains responses, deserializes `predict_action_chunk` dicts, and pushes them into the `ActionQueue` exactly the way `AsyncExecution(transport="process")` does.
- `warmup` triggers a server-side warmup RPC that runs `n` warmup inferences on the server's worker before the first real request.
- `estimated_delay_frames()` reports the rolling end-to-end latency observed by the client (network + server compute), so RTC's prefix mask remains correct.

`ActionQueue`, `ChunkSmoother`, `RTCQueueMerger`, and all callbacks remain on the robot host and are unchanged. From the runtime's perspective, `RemoteExecution` is just another `Execution`.

---

## 4. Server-Host Responsibilities (`PolicyServer`)

`PolicyServer` is a process in `physicalai.runtime.execution.remote.server`. Its responsibilities:

- **Model lifecycle.** Load `InferenceModel` from a manifest, run startup warmup, expose `replace_model` and `unload` RPCs for hot-reload.
- **Request handling.** Implement `predict_action_chunk` and `select_action` RPCs that delegate to the loaded model. Reject requests during model swap; reject requests with a schema version that does not match the loaded model.
- **Guidance compatibility.** RTC and other `Guidance` implementations live on the server (they are part of the runner). The client sends `inference_delay` and `prev_chunk_left_over` over the wire (see §6); the server-side runner consumes them inside the denoising loop.
- **Observability.** Per-request structured logs (request id, observation timestamp, server compute time, queue depth on the server). A `/healthz` endpoint for orchestration. No PhysicalAI-side metrics framework is mandated; the server emits structured records that the deployment's existing collector consumes.
- **Concurrency.** One `InferenceModel` per server process. Multi-robot serving is achieved by running one `PolicyServer` per robot client (or a small pool with a request router). Sharing one model across concurrent requests requires per-request state isolation (resetting the runner's chunk history, for example) that the initial design does not attempt to provide. This restriction is documented; multi-robot single-server is left to a follow-up (see §10).

---

## 5. Transport and Framing

Recommended transport: **gRPC over HTTP/2 with bidirectional streaming**. Justification:

- The control loop is a long-lived bidirectional stream of (observation snapshot → action chunk) pairs at a fixed rate. Bidirectional streaming amortizes connection setup across the entire session.
- gRPC's deadline propagation gives `RemoteExecution` a clean cancellation primitive when the runtime stops mid-request.
- Protobuf gives a typed schema for the `predict_action_chunk` dict (which is dict-shaped on the Python side but benefits from a wire schema for cross-version validation).
- gRPC client libraries support async I/O cleanly, which the reader thread / async task on the client side relies on.

Open alternative: HTTP/2 + JSON for the control plane (manifest exchange, warmup, health) plus a separate stream for inference. Recorded as §10.2 question 1.

Wire schema (proto, sketch):

```proto
service PolicyServer {
  rpc Handshake(HandshakeRequest) returns (HandshakeReply);
  rpc Warmup(WarmupRequest) returns (WarmupReply);
  rpc Predict(stream PredictRequest) returns (stream PredictReply);
  rpc Health(google.protobuf.Empty) returns (HealthReply);
}

message PredictRequest {
  string request_id = 1;
  double t0 = 2;                              // observation timestamp
  map<string, Tensor> observation = 3;        // typed serialization of dict observation
  optional int32 inference_delay = 4;         // RTC: in policy frames
  optional ActionChunk prev_chunk_left_over = 5;  // RTC: leftover from prior chunk
}

message PredictReply {
  string request_id = 1;
  double t0 = 2;
  Tensor actions = 3;                         // (H, D)
  optional double policy_dt = 4;
  map<string, Tensor> extra = 5;              // unknown keys passed through
}
```

`Tensor` is a small message carrying dtype, shape, and bytes. Image observations cross the wire as raw bytes plus dtype/shape; compression (JPEG/PNG/AV1) is a configurable codec on the client side, applied before serialization, decoded on the server. Codec choice is part of the handshake.

---

## 6. RTC Compatibility

RTC works over the wire without changes to the `Guidance` API:

- The client (`RemoteExecution`) tracks `estimated_delay_frames()` from observed end-to-end latency exactly as `AsyncExecution` does. It computes `inference_delay` and includes it in `PredictRequest`.
- The client also tracks the previous chunk's leftover actions (the same way `AsyncExecution` does today, since RTC overlap state is `Execution`-side, not model-side) and includes `prev_chunk_left_over` in the request.
- The server-side runner is `FlowMatching(guidance=RTC())`. It receives `inference_delay` and `prev_chunk_left_over` as kwargs to `predict_action_chunk`, exactly as the local async path does.

The constraint this imposes: `prev_chunk_left_over` must serialize cleanly. Since it is a dict of arrays, it goes through the same `Tensor` serialization as observations. No additional protocol surface is needed.

Relative-action reanchoring (`policy_runtime_design.md` §8.1) happens on the server because the snapshot state used for reanchoring is part of the request's `observation` payload. No client-side reanchoring logic is needed.

---

## 7. Failure Modes and Policy

Network transports introduce failure modes that local transports do not. `RemoteExecution` owns the policy for each:

| Failure | Behavior |
|---|---|
| Connection loss before first request | `start()` raises; `PolicyRuntime.run()` exits cleanly through the `finally` block |
| Connection loss mid-stream, no request in flight | reconnect with backoff up to `reconnect_budget_s`; if exhausted, raise on the next `maybe_request` |
| Connection loss with request in flight | the in-flight request is considered lost; reconnect; do not retry the lost request (the observation is stale by the time reconnection completes); next `maybe_request` continues normally |
| Server-side model swap mid-stream | server returns a typed `MODEL_SWAPPED` error; client raises the error to the runtime, which exits cleanly |
| Server-side request error (validation, OOM, runner exception) | server returns the error in `PredictReply`; client raises on the reader thread; surfaces on next runtime call |
| Deadline exceeded | request is dropped; `estimated_delay_frames()` updates from the timeout, not from a successful response; next `maybe_request` continues |
| Schema mismatch at handshake | `start()` raises with both versions in the error message; runtime never enters the loop |

The "do not retry the lost request" rule mirrors `AsyncExecution`'s no-cancel-and-no-retry default (`policy_runtime_design.md` §20, Open Question 2). Retries are not the right primitive for control-loop inference: by the time a retried response arrives, the observation it was computed from is stale.

---

## 8. Configuration Surface

`RemoteExecution`'s constructor (sketch):

```python
class RemoteExecution(Execution):
    def __init__(
        self,
        endpoint: str,                          # "grpc://host:port" or "https://host:port"
        *,
        auth: AuthConfig | None = None,         # mTLS material, bearer token, or None
        request_deadline_s: float = 1.0,        # per-request deadline
        reconnect_budget_s: float = 10.0,       # total time to spend reconnecting
        image_codec: ImageCodec = ImageCodec.JPEG,  # client-side compression
        refill_threshold: int | None = None,    # same semantics as AsyncExecution
    ): ...
```

Manifest example:

```yaml
runtime:
  class_path: physicalai.runtime.PolicyRuntime
  init_args:
    fps: 30
    execution:
      class_path: physicalai.runtime.execution.RemoteExecution
      init_args:
        endpoint: grpc://gpu-host-1:50051
        auth:
          class_path: physicalai.runtime.execution.remote.MutualTLSAuth
          init_args:
            client_cert: /etc/policy-client/cert.pem
            client_key:  /etc/policy-client/key.pem
            ca_cert:     /etc/policy-client/ca.pem
        request_deadline_s: 0.5
        image_codec: jpeg
```

The server is launched separately:

```bash
physicalai serve --config policy_server.yaml
```

`physicalai serve` is a new CLI subcommand parallel to `physicalai run`. It loads `InferenceModel` from a manifest, runs warmup, and binds the gRPC service. The subcommand is registered as a runtime-distribution entry point under the `physicalai.cli.subcommands` group (`policy_runtime_design.md` §14): the runtime CLI lives in `physicalai/src/physicalai/cli/`, uses plain `jsonargparse` (no Lightning, no Torch), and discovers `serve` through `importlib.metadata.entry_points`. The handler lazy-imports `physicalai.runtime.execution.remote.server` so the gRPC and server-side dependencies are paid only when `serve` is actually invoked.

---

## 9. Differences from LeRobot

LeRobot's `PolicyServer` / `RobotClient` design is the closest reference. The differences here are deliberate:

| LeRobot | This design | Why |
|---|---|---|
| `RobotClient` is a peer object alongside `InferenceEngine` | `RemoteExecution` is an `Execution` implementation | reuses the `Execution` contract that `Sync` / `Async` already satisfy; `PolicyRuntime` does not learn about "remote" as a special case |
| `PolicyServer` exposes the inference engine API directly | `PolicyServer` exposes `predict_action_chunk` / `select_action` and runs the runner / guidance internally | client side stays free of runner / guidance dependencies; only the model interface crosses the wire |
| RTC overlap state handled inside `RTCInferenceEngine` end-to-end | RTC overlap state crosses the wire as `prev_chunk_left_over` in the request payload | matches the local async path; the server-side `Guidance` is the same class as the local one |
| Server is colocated with rollout strategies | Server is a runtime-only component; strategies stay in the consumer (`policy_runtime_design.md` §19) | preserves the separation: the loop and the model live in `physicalai`; product workflows live in the consumer |

---

## 10. Out of Scope and Open Questions

### 10.1 Out of scope

Listed so they are not assumed:

- **Multi-tenant single server.** One `InferenceModel` per `PolicyServer` process. Multi-robot serving is a follow-up.
- **Server-side action smoothing.** `ChunkSmoother` and `ActionQueue` stay on the robot host. The server returns raw chunks.
- **Server-side robot state.** The server is stateless across requests except for the runner's internal state (chunk history, denoising step state). A failed connection drops that state; the next request starts fresh.
- **Privileged operations over the wire.** Hot-reload, model swap, and model unload are admin RPCs guarded by `auth`. They are not exposed to the inference path.
- **Latency optimization beyond gRPC defaults.** Shared-memory tensors over local network, RDMA, and similar optimizations are deferred until a deployment actually needs them.

### 10.2 Open questions

1. **Transport choice.** §5 recommends gRPC bidirectional streaming. Alternatives: HTTP/2 + JSON for the control plane plus a separate inference stream; raw TCP with a custom framing for the lowest-latency local-network case. Revisit once a deployment commits to a specific orchestration stack and observability stack.
2. **Multi-tenancy.** §10.1 lists multi-robot single-server as out of scope. Revisit when a deployment runs more robots than affordable GPU hosts, and per-robot model state isolation can be made cheap (most likely by running multiple model instances inside one process rather than sharing one model across requests).
3. **Authentication posture.** Default in §8 is mutual TLS. Bearer-token and unauth (loopback / trusted-network) modes are supported through the `AuthConfig` interface but not specified here in detail.

---

## 11. File Layout

```text
physicalai/runtime/execution/remote/
  __init__.py            # RemoteExecution (client-side Execution implementation)
  client.py              # gRPC client, reader thread, latency tracking
  server.py              # PolicyServer process (loads InferenceModel; serves Predict)
  proto/                 # generated gRPC stubs from policy_server.proto
  auth.py                # AuthConfig, MutualTLSAuth, BearerTokenAuth
  codecs.py              # ImageCodec (JPEG/PNG/AV1) + Tensor (de)serialization
```

The server module is imported lazily by the `physicalai serve` subcommand to keep the dependency footprint of the main runtime untouched on robot hosts that never run a server.

---

## 12. Build Sequencing

Phase 5 acceptance criteria for `RemoteExecution` and `PolicyServer` live in `policy_runtime_design.md` §15. They are not duplicated here. Anything that changes the acceptance criteria should be reflected in both documents.

---

## 13. References

- `policy_runtime_design.md` — `Execution` contract (§10), Phase 5 acceptance criteria (§15), studio strategy composition (§19).
- LeRobot `PolicyServer` / `RobotClient` — closest reference; differences in §9.
