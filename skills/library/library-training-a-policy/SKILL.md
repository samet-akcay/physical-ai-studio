---
name: library-training-a-policy
description: Trains, validates, tests, and runs prediction for Physical AI Studio policies through the Lightning-based CLI. Use when running physicalai fit/validate/test/predict, writing or editing YAML configs under library/configs, wiring a model + datamodule + trainer, resuming from a checkpoint, or debugging a training run. Covers ACT, Pi0, Pi0.5, GR00T, and SmolVLA.
license: Apache-2.0
---

# Training a Studio Policy

Training runs through `physicalai fit` (and its siblings `validate`, `test`, `predict`), which drive `physicalai.train.Trainer` — a subclass of `lightning.Trainer` in `library/src/physicalai/train/trainer.py`. Runs are configured with jsonargparse YAML in `library/configs/` and write to `experiments/{name}/version_N/` by default.

The four Lightning subcommands share the same `--model` / `--data` / `--trainer.*` shape (see `cli/_dispatch.py`); `validate`/`test`/`predict` additionally take `--ckpt_path`.

## Anatomy of a config

A config wires three pieces via `class_path` / `init_args`:

- `model` — a `Policy` subclass (e.g. `physicalai.policies.ACT`).
- `data` — a `DataModule`, usually `physicalai.data.lerobot.LeRobotDataModule` with a `repo_id` (e.g. `lerobot/pusht`).
- `trainer` — Lightning args (`max_epochs`, `accelerator`, `devices`, callbacks…).

Configs live in `library/configs/physicalai/` (first-party: `act.yaml`, `pi0.yaml`, `pi05.yaml`, `groot.yaml`, `smolvla.yaml`) and `library/configs/lerobot/` (LeRobot-wrapped). Compose with `__base__` and override any field on the CLI (`--trainer.max_epochs 200 --data.train_batch_size 64`).

## Workflow

1. **Start from an existing config** matching your policy family; copy it rather than writing from scratch.
   - Done when: `physicalai fit --config <your.yaml> --print_config` renders the fully-resolved config with no errors.
2. **Smoke-test the wiring** before a real run:
   ```bash
   physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
   ```
   - Done when: one train + one val batch complete without shape or config errors.
3. **Run training**, overriding on the CLI as needed:
   ```bash
   physicalai fit --config configs/physicalai/<name>.yaml --trainer.max_epochs 200
   ```
   - Done when: checkpoints appear under `experiments/{name}/version_N/`.
4. **Validate / test / predict** from a checkpoint:
   ```bash
   physicalai validate --config configs/physicalai/<name>.yaml --ckpt_path experiments/<name>/version_0/checkpoints/last.ckpt
   ```
5. **Iterate on metrics**, not just loss — confirm the val metric relevant to the task moves, and record the config + checkpoint that produced it.

## Debugging a run

- `--trainer.fast_dev_run=true` — one batch each stage; the first thing to try on any failure.
- `--print_config` — see the exact resolved config jsonargparse built.
- Shape/feature mismatches usually mean the datamodule's `Feature` names or action dim disagree with the policy — cross-check against the `library-adding-a-policy` skill.
- Dataset download stalls: the run is pulling a LeRobot `repo_id`; see the `library-working-with-datasets` skill.

## Required checks

- Config resolves (`--print_config`) and `fast_dev_run` passes before any long run.
- `accelerator`/`devices` match the installed backend extra (`xpu`/`cuda`/`cpu`).
- New or renamed config fields stay consistent with the policy's `Config` class.
- Doc code blocks that show training commands still pass `tests/test_docs.py`.

## Verify

```bash
# from library/
physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
uv run pytest tests/unit/train
```

## Related skills

- `library-adding-a-policy` — when the model itself needs changes.
- `library-working-with-datasets` — for the `data` half of the config.
- `library-benchmarking-a-policy` — to evaluate a trained checkpoint in a gym.
