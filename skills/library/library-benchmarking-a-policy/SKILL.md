---
name: library-benchmarking-a-policy
description: Benchmarks a trained Physical AI Studio policy in a simulation gym and reports success metrics. Use when running physicalai benchmark, editing configs under library/configs/benchmark, adding or changing a Benchmark class in physicalai.benchmark, tuning rollout/episode/env settings, recording rollout videos, or interpreting results.json / results.csv.
license: Apache-2.0
---

# Benchmarking a Studio Policy

`physicalai benchmark` evaluates a trained policy by rolling it out in a gym and scoring success. Benchmark classes live in `library/src/physicalai/benchmark/gyms/benchmark.py` (`Benchmark`, `PushTBenchmark`, `LiberoBenchmark`); results types in `benchmark/gyms/results.py` (`BenchmarkResults`, `TaskResult`); rollout logic in `library/src/physicalai/eval/rollout.py` (`evaluate_policy`). CLI entry: `library/src/physicalai/cli/benchmark.py`.

## Invocation

```bash
physicalai benchmark \
  --config configs/benchmark/pusht.yaml \
  --policy physicalai.policies.ACT \
  --ckpt_path experiments/act/version_0/checkpoints/last.ckpt \
  --output_dir ./results/benchmark
```

- `--policy` — policy class path.
- `--ckpt_path` — a `.ckpt` **or** an export directory.
- `--config` — a benchmark config (`configs/benchmark/pusht.yaml`, `configs/benchmark/libero.yaml`) selecting the `Benchmark` class and its settings.
- `--output_dir` — defaults to `./results/benchmark`.

Override benchmark settings on the CLI, e.g. `--benchmark.num_episodes 10 --benchmark.num_envs 8`.

## Output

- Prints `results.summary()` to stdout.
- Writes `results.json` and `results.csv` into `--output_dir`.
- Optional video via config `video_dir` + `record_mode` (`all` | `failures` | `successes` | `none`).

## Workflow

1. **Confirm the policy loads** from the checkpoint/export before a full sweep:
   ```bash
   physicalai benchmark --config configs/benchmark/<suite>.yaml --policy <ClassPath> --ckpt_path <path> --benchmark.num_episodes 1
   ```
   - Done when: one episode runs end-to-end and a summary prints.
2. **Run the full benchmark** with the intended episode/env counts.
   - Done when: `results.json` and `results.csv` are written and the success metric is populated.
3. **Interpret results** via `BenchmarkResults`/`TaskResult` fields; compare against a baseline checkpoint on the same config.
4. **Record videos** for qualitative review when a task regresses (`record_mode: failures`).

## Adding or changing a Benchmark

1. Subclass `Benchmark` in `benchmark/gyms/` (study `PushTBenchmark` / `LiberoBenchmark`); the gym itself comes from `physicalai.gyms` (`pusht.py`, `libero.py`, …).
2. Add a matching config in `library/configs/benchmark/`.
3. Add tests under `library/tests/unit/benchmark/`.
   - Done when: `uv run pytest tests/unit/benchmark` passes and a 1-episode run succeeds.

## Required checks

- The policy runs from **both** a `.ckpt` and an export dir if both are supported paths.
- Success/episode metrics are populated (not zero/NaN by accident) and reproducible across runs.
- Env/episode counts match hardware; large `num_envs` fits memory.
- Heavy gym deps (e.g. `libero`, `robocasa`) are gated behind their optional extras and imported lazily.

## Related skills

- `library-training-a-policy` — to produce the checkpoint being benchmarked.
- `library-exporting-and-validating` — when benchmarking an exported artifact for deployment parity.
