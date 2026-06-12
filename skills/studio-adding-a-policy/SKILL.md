---
name: studio-adding-a-policy
description: Add or modify a Physical AI Studio policy implementation. Use when creating a policy under physicalai.policies, wiring policy config/model/policy classes, adding CLI/config registration, tests, docs, or ensuring training/export compatibility for ACT, Pi0, Pi0.5, GR00T, SmolVLA, LeRobot, or a new policy family.
license: Apache-2.0
---

# Adding a Studio Policy

Use this skill when adding or changing policy code in `library/src/physicalai/policies`.

## Workflow

1. Inspect nearby policy implementations before writing code. Prefer the smallest change that matches existing package structure.
2. Keep the policy split explicit: `config.py` for configuration, `model.py` for `torch.nn.Module` code, and `policy.py` for the Lightning `Policy` wrapper.
3. Implement the policy interface used by training and inference: `forward(...)`, `predict_action_chunk(...)`, `select_action(...)` behavior through the base policy, and checkpoint/config construction patterns.
4. Wire exports only when the policy can provide a valid sample input and backend-specific constraints are understood.
5. Add or update tests near existing policy tests. Include shape validation and at least one construction/config path.
6. Update docs or examples if the new policy is user-visible.

## Required Checks

- Confirm action shape semantics: batch, horizon/chunk length, and action dimension.
- Confirm observation feature names align with dataset/config conventions.
- Confirm config loading works through the CLI-facing config path if the policy is meant to be used with `physicalai fit`.
- Confirm export support is either implemented and tested or explicitly documented as unsupported.

## References

- See `references/base-classes.md` for the policy class contract and existing patterns.
