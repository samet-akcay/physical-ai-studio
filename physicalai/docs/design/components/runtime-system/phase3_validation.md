# Phase 3 Hardware Validation

**Date**: 2026-05-14
**Hardware**: SO-101 (`/dev/ttyACM0`, damaged joints 2 & 3, ResilientSO101 wrapper) + UVC camera (`/dev/video0`)
**Policy**: MolmoAct2 (bfloat16, CUDA), exported via `physicalai.export(backend="torch")`
**Task prompt**: "pick up the yellow cube"
**Calibration**: `/home/sakcay/.cache/physicalai/robots/8d8353a0-fc8c-49aa-b5f1-33290f726698/calibrations/6e6303f3-495a-4e75-8d30-edcc8932c7dd.json`

Both runs used `scripts/policy_runtime_dryrun.py` with `--bfloat16 --resilient`. Robot does not actuate (DryRunInterceptor echoes state as action).

## Results

| Metric | Sync (`SyncInferenceExecution`, mode="chunk") | Async (`AsyncInferenceExecution`, refill_threshold=2) |
|---|---|---|
| Duration | 10 s | 60 s |
| FPS target | 2.0 Hz | 2.0 Hz |
| Total ticks | 17 | **120** |
| **Effective Hz** | **1.7 Hz (15 % below target)** | **2.0 Hz (on target)** |
| Tick wall-clock mean | 598 ms | 501 ms |
| Tick wall-clock p50 | 513 ms | 532 ms |
| Tick wall-clock **p99** | **1774 ms** | **982 ms** |
| Tick wall-clock max | 1966 ms | 1035 ms |
| Inferences | ~2 (chunk re-inference) | 119 (one per tick) |
| Inference latency mean | n/a | 23 ms |
| Inference latency p99 | n/a | 239 ms |
| Transient failures | 0 | 0 |
| Fatal failures | 0 | 0 |
| Worker shutdown clean | n/a | yes |
| Warmup pre-fill | 1 inference | 2 inferences |
| Fallback invocations | n/a | 0 (warmup pre-filled queue) |

> Tick wall-clock includes the `1/fps = 500 ms` sleep interval; the work-only portion is roughly `tick - 500 ms`.

## Interpretation

- **Sync** drops below target FPS (1.7 Hz vs 2.0 Hz) because chunk re-inference blocks the loop for ~1.5 s every chunk boundary (p99 = 1774 ms).
- **Async** hits target FPS (2.0 Hz) consistently. Worst-case tick is ~982 ms vs sync's 1774 ms — a **~45 % p99 reduction**.
- Inference latency p99 (239 ms) is well under the 500 ms tick budget, leaving headroom for higher FPS.
- No transient or fatal failures over 60 s and 119 inferences.

## Phase 3 acceptance

- ✅ `FallbackAction` integrated (no fallback invocations needed because warmup pre-filled the queue).
- ✅ `AsyncInferenceExecution` ran 60 s clean with 119 inferences, no failures, clean shutdown.
- ✅ `runtime.warmup()` pre-filled the queue and avoided fallback at tick 1.
- ✅ Generic `policy_runtime_dryrun.py` works with MolmoAct2 today; CLI is policy-agnostic.

## Bug fixes during validation

1. **`_build_sample_with_task` retried 10×** — first servo read on damaged SO-101 frequently fails; ResilientSO101 needs `_last_good_obs` populated before it can mask errors, so the warmup sample reader must retry independently.
2. **`RobotRuntime.warmup()` now starts the controller** — `AsyncInferenceExecution.warmup()` requires `start()` first; runtime tracks `_controller_started` and calls `start()` on demand from either `warmup()` or `run()`.

## Reproduction

```bash
# Sync baseline (10 s)
library/.venv/bin/python scripts/policy_runtime_dryrun.py \
  --export-dir /tmp/molmoact2_bf16 --bfloat16 \
  --robot so101 --robot-port /dev/ttyACM0 \
  --robot-calibration /home/sakcay/.cache/physicalai/robots/8d8353a0-fc8c-49aa-b5f1-33290f726698/calibrations/6e6303f3-495a-4e75-8d30-edcc8932c7dd.json \
  --resilient --camera uvc --camera-index 0 \
  --execution sync --warmup 1 --fps 2 --duration 10 \
  --task "pick up the yellow cube"

# Async validation (60 s)
library/.venv/bin/python scripts/policy_runtime_dryrun.py \
  --export-dir /tmp/molmoact2_bf16 --bfloat16 \
  --robot so101 --robot-port /dev/ttyACM0 \
  --robot-calibration /home/sakcay/.cache/physicalai/robots/8d8353a0-fc8c-49aa-b5f1-33290f726698/calibrations/6e6303f3-495a-4e75-8d30-edcc8932c7dd.json \
  --resilient --camera uvc --camera-index 0 \
  --execution async --warmup 2 --refill-threshold 2 --fps 2 --duration 60 \
  --task "pick up the yellow cube"
```
