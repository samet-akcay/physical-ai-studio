---
name: library-working-with-datasets
description: Works with Physical AI Studio datasets and Lightning datamodules built on the LeRobot format. Use when wiring physicalai.data.lerobot.LeRobotDataModule into a training config, choosing a repo_id, converting between the physicalai and lerobot data layouts, defining observation Features/FeatureType, setting normalization, or debugging batch shapes and dataloading.
license: Apache-2.0
---

# Working with Studio Datasets

Studio data lives in `library/src/physicalai/data/`. Datasets use the **LeRobot format** and are consumed through Lightning datamodules.

Key modules:

- `data/lerobot/datamodule.py` — `LeRobotDataModule` (the class configs reference as `physicalai.data.lerobot.LeRobotDataModule`).
- `data/lerobot/dataset.py` — LeRobot dataset wrapper.
- `data/lerobot/converters.py` — `DataFormat` (StrEnum: `physicalai`, `lerobot`) and bidirectional field mapping between the two layouts.
- `data/observation.py` — `Observation`, `Feature`, `FeatureType`, `NormalizationParameters`.
- `data/datamodules.py` — base `DataModule` (Lightning `LightningDataModule`, auto num-workers heuristic).
- `data/dataset.py` — base `Dataset`; `data/gym.py` — `GymDataset` for gym-generated data.

## Wiring data into a training config

In a `physicalai fit` config, the `data` block selects the datamodule and its `repo_id`:

```yaml
data:
  class_path: physicalai.data.lerobot.LeRobotDataModule
  init_args:
    repo_id: lerobot/pusht
    train_batch_size: 64
```

`repo_id` points at a LeRobot/HuggingFace dataset; the datamodule pulls it on first use. See the `library-training-a-policy` skill for the full config.

## Workflow

1. **Pick the dataset** by `repo_id` and confirm its features (image keys, state dim, action dim) match the target policy's `Config`.
   - Done when: the policy's expected `Feature` names and action dimension line up with the dataset.
2. **Verify a batch** before training:
   ```bash
   physicalai fit --config <config.yaml> --trainer.fast_dev_run=true
   ```
   - Done when: one batch flows through with correct shapes and no missing-feature errors.
3. **Convert layouts** only when needed via `converters.py` (`DataFormat.physicalai` ↔ `DataFormat.lerobot`); keep field names stable, since they propagate to training and export.
4. **Set normalization** through `NormalizationParameters`/`Feature` consistently with what the policy expects at inference.

## Debugging dataloading

- Missing/renamed feature → the config's dataset features disagree with the policy; align `Feature` names in `data/observation.py` conventions.
- Slow/stalled first batch → the LeRobot `repo_id` is downloading; expected on first run (see the `requires_download` test marker for tests that need this).
- Wrong batch dimensions → check `train_batch_size` and the datamodule's collate/observation handling before changing the policy.

## Required checks

- Feature names, `FeatureType`, action dim, and normalization match between dataset, `Config`, and any export metadata.
- Conversions round-trip without dropping or renaming fields.
- Tests that require downloads are marked `requires_download`; keep default `uv run pytest` runnable offline.

## Verify

```bash
# from library/
uv run pytest tests/unit/data tests/unit/datamodules
```

## Related skills

- `library-training-a-policy` — the `data` block is one half of a training config.
- `library-adding-a-policy` — align observation features with the policy `Config`.
