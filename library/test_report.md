# MolmoAct2 LeRobot Wrapper - Test Report

**Branch**: `feat/add-molmo-act-2`

## Why integration felt “stuck”

1. **Quiet + pipes** — Runs like `pytest tests/integration -q 2>&1 | tail -30` buffer all output until pytest exits, so nothing appears for 30–60+ minutes even when tests are progressing.
2. **Heavy parametrization** — Each VLA policy (pi0, pi05, smolvla, molmoact2, …) loads multi‑GB weights. One full `test_lerobot_wrapper_e2e.py` pass is ~12 minutes on GPU.
3. **Old equivalence bug (fixed)** — Tests used `xfail` for `molmoact2` / `groot` / `xvla` but **still executed** them (Hub download + train). That added tens of minutes for no value. Equivalence now only runs `VALIDATED_EQUIVALENCE_POLICIES` (6 policies).

**CI** (`.github/workflows/library.yml`) runs **`tests/unit` only**, not full integration.

## How to run with visible progress

From `library/` with your lerobot env active:

```bash
# Unit (~10 min)
pytest tests/unit -v --durations=10

# MolmoAct2 export round-trip only (~40s)
pytest tests/integration/test_lerobot_wrapper_e2e.py -v -k molmoact2 --durations=5

# All wrapper export e2e (~12 min GPU)
pytest tests/integration/test_lerobot_wrapper_e2e.py -v --durations=10

# Equivalence (non-slow tiers, 6 validated policies — still GPU-heavy for VLAs)
pytest tests/integration/test_lerobot_wrapper_equivalence.py -m "not slow" -v --durations=15
```

Avoid `-q` and avoid piping to `tail`/`head` unless you use `pytest --capture=tee-sys` or wait for completion.

## Latest local results

| Suite | Result |
|-------|--------|
| Unit | **645 passed**, 5 skipped (~10 min) |
| E2E wrapper (molmoact2 only) | **PASSED** (~37 s, verbose) |
| Full e2e (all policies) | 7 pass, 2 xfail (groot, xvla) when run end-to-end (~12 min) |

## Library changes (summary)

- **`_coerce_policy_config_kwargs`**: `dtype` → `model_dtype` / module cast for VLAs.
- **Thin `MolmoAct2`**: deployment via `MolmoAct2.from_config(MolmoAct2Config(...))` (see `scripts/molmoact2_dry_run.py`).
- **E2E test kwargs**: export scaffold flags in `test_lerobot_wrapper_e2e._get_policy_kwargs` only.
- **Equivalence scope**: only `VALIDATED_EQUIVALENCE_POLICIES` (excludes molmoact2; use export e2e for MolmoAct2).
