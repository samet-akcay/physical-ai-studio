---
name: library-adding-a-policy
description: Adds or modifies a Physical AI Studio policy under library/src/physicalai/policies. Use when creating a new policy family with the config/model/policy split, registering it in the get_policy factory and package exports, or keeping a policy compatible with Lightning training and export. Covers ACT, Pi0, Pi0.5, GR00T, SmolVLA, and LeRobot-wrapped policies.
license: Apache-2.0
---

# Adding a Studio Policy

Policies live in `library/src/physicalai/policies/<name>/`. Each family is a Lightning-facing `Policy` wrapping a `torch.nn.Module` `Model`, split across three files. Base classes are in `policies/base/` (`Policy` in `policy.py`, `Model` in `model.py`); the config base `Config` is in `library/src/physicalai/config/`.

## Workflow

1. **Read a nearby family first.** Study `policies/act/` (smallest complete example): `config.py` (`ACTConfig(Config)`), `model.py` (`ACT(Model)`, the nn.Module), `policy.py` (`ACT(ExportablePolicyMixin, Policy)`), `preprocessor.py`. Make the smallest change that matches this structure.
   - Done when: you can name which existing file each new file mirrors.
2. **Create the three-file split** in `policies/<name>/`:
   - `config.py` — `<Name>Config(Config)`, all hyperparameters as typed fields.
   - `model.py` — `<Name>Model(Model)`, pure `torch.nn.Module` logic.
   - `policy.py` — `<Name>(Policy)` (add `ExportablePolicyMixin` only when export is implemented).
   - Done when: `from physicalai.policies.<name> import <Name>, <Name>Config, <Name>Model` imports cleanly.
3. **Implement the policy interface** used by both training and inference through the base `Policy`:
   - `forward(...)` — training path; return values compatible with `training_step`.
   - `predict_action_chunk(...)` — inference path; return a tensor with the configured action horizon.
   - `select_action(...)` — use base-class action-queue behavior unless a specialized flow is justified.
   - Done when: shapes match the checks below for a synthetic batch.
4. **Register the family** so the CLI and factory can find it:
   - Add exports to `policies/__init__.py` (`__all__` and imports, e.g. `<Name>`, `<Name>Config`, `<Name>Model`).
   - Add the lowercase name to the `get_physicalai_policy_class(...)` / `get_policy(...)` dispatch in `policies/__init__.py`.
   - Done when: `get_policy("<name>")` returns the class and `--model physicalai.policies.<Name>` resolves.
5. **Add a training config** in `library/configs/physicalai/<name>.yaml` wiring `model.class_path`, a `data.class_path` (usually `physicalai.data.lerobot.LeRobotDataModule`), and `trainer.*`. Mirror `configs/physicalai/act.yaml`.
   - Done when: `physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true` completes one step.
6. **Wire export only when ready.** Add `ExportablePolicyMixin` and a valid sample input, then follow the `library-exporting-and-validating` skill. If export is intentionally unsupported, say so explicitly in the policy docstring.
7. **Add tests** under `library/tests/unit/policies/` next to existing policy tests: at least one construction/config path and one shape-validation test.
   - Done when: `uv run pytest tests/unit/policies -k <name>` passes.
8. **Update docs** if the policy is user-visible: `library/docs/explanation/policy/` and any config examples.

## Required checks

Account for every item below (not just "looks fine"):

- **Action shape semantics** — batch, horizon/chunk length, and action dimension are correct and unchanged from the family's convention.
- **Observation features** — feature names align with dataset/config conventions (`data/observation.py`: `Feature`, `FeatureType`).
- **Config path** — construction works through the jsonargparse CLI path used by `physicalai fit` (`class_path`/`init_args`).
- **Heavy dependencies** — gate large families behind an optional extra in `library/pyproject.toml` and import lazily, matching `pi0`/`groot`/`smolvla`.
- **No silent contract changes** — do not alter action dims, feature names, or preprocessing without coordinating export/Runtime.

## Verify

From `library/`:

```bash
uv run pytest tests/unit/policies -k <name>
physicalai fit --config configs/physicalai/<name>.yaml --trainer.fast_dev_run=true
prek run --all-files library/
```

## References

- `references/base-classes.md` — the `Policy`/`Model` contract and file-split expectations.
