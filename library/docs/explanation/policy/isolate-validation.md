# Runtime isolate and policy-plugin validation

This document is a cold-reader checklist and records opt-in evidence for the Runtime isolate integration and the Studio LeRobot/GR00T plugins. It validates the current branch stack; it does not claim that every framework workflow or deployment target is supported.

## Stack under test

- Runtime: [`samet-akcay/physicalai#8`](https://github.com/samet-akcay/physicalai/pull/8), `feat/physicalai-isolate`, resolved commit `c26a96952d146a1f54d21a0e9edcd8cbd5f099d2`.
- Studio: [`samet-akcay/physical-ai-studio#10`](https://github.com/samet-akcay/physical-ai-studio/pull/10) and [`#23`](https://github.com/samet-akcay/physical-ai-studio/pull/23). The validation branch starts at `feat/gr00t-engine-plugin` commit `15a1786135d83c334b87119db1c02e034b8a983d`.
- `library/pyproject.toml` points `physicalai` at `feat/physicalai-isolate`; `library/uv.lock` resolves that branch to the Runtime commit above.
- The GR00T plugin pins Isaac-GR00T commit `3df8b3825d67f755e69141446f4315f281b9b7e6`. Its OpenVINO environment uses Robotics AI Suite revision `2960959ae6cc6d45150316ed6612f686c2d915b7`, eight published patches, and resulting tree `d657bcb89b8937495b8c849d7cfb8c9f3d1e5601`.

## Host and dependencies

Validation ran on 2026-10-23 on Ubuntu 24.04, host `HP-01-LAB5002-L-R1`:

| Component     | Version / value                                                            |
| ------------- | -------------------------------------------------------------------------- |
| GPU           | NVIDIA RTX A6000, 49,140 MiB                                               |
| NVIDIA driver | 580.178.04 (driver reports CUDA 13.0)                                      |
| Host Python   | 3.12.3                                                                     |
| Host `uv` CLI | 0.10.4                                                                     |
| Host PyTorch  | 2.11.0+cu128; `torch.cuda.is_available()` returned true; CUDA runtime 12.8 |
| LeRobot       | 0.6.1                                                                      |
| Studio SHA    | `15a1786135d83c334b87119db1c02e034b8a983d`                                 |
| Runtime SHA   | `c26a96952d146a1f54d21a0e9edcd8cbd5f099d2`                                 |

Prepare the Studio test environment from `library/`:

```bash
uv lock --check
uv sync --locked --extra cu128 --extra tests --extra lerobot --extra gr00t --no-default-groups
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
```

The GR00T end-to-end test needs access to its pinned public inputs, including `nvidia/Cosmos-Reason2-2B`. Keep Hugging Face credentials in the host credential store; do not paste tokens into commands, test output, or this document. The Runtime-only host below installs the plugin without the `train` extra and does not install `physicalai-train`.

Environment keys used for the recorded runs:

| Check                          | Environment variables                                                                                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1 Runtime-only                | No `PHYSICALAI_*` overrides; `PHYSICALAI_ISOLATE_CACHE` was unset, so Runtime used its default cache.                                               |
| B LeRobot training             | `PHYSICALAI_RUN_LEROBOT_INTEGRATION=1`, `PHYSICALAI_LEROBOT_TEST_STEPS=5`, `PHYSICALAI_LEROBOT_DEVICE=cuda`                                         |
| B LeRobot pretrained inference | `PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE=1`, `PHYSICALAI_LEROBOT_DEVICE=cuda`                                                                   |
| B3 GR00T end-to-end            | `PHYSICALAI_RUN_GR00T_E2E=1`, `PHYSICALAI_GR00T_E2E_OUTPUT=/tmp/isolate-validation-gr00t-e2e-fixed`; step/batch settings used test defaults (20/8). |
| C GR00T plugin tests           | No test opt-in variables.                                                                                                                           |

## Checklist

### A1 — Runtime-only GR00T import, prepare, and infer

Create a separate environment with only Runtime and the GR00T plugin. This catches accidental imports of `physicalai.engine` and proves the plugin does not require `physicalai-train` for deployment.

```bash
repo_root="$(git rev-parse --show-toplevel)"
runtime_env="$(mktemp -d)"
cat >"$runtime_env/pyproject.toml" <<EOF
[project]
name = "runtime-only-gr00t-validation"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["physicalai", "physicalai-gr00t-policy-plugin"]

[tool.uv.sources]
physicalai = { git = "https://github.com/samet-akcay/physicalai", branch = "feat/physicalai-isolate" }
physicalai-gr00t-policy-plugin = { path = "$repo_root/plugins/policies/gr00t" }
EOF
uv sync --project "$runtime_env" --no-default-groups
uv tree --project "$runtime_env"
```

The tree must not contain `physicalai-train`. Import the Runtime-safe declarations and prepare both managed environments:

```bash
uv run --project "$runtime_env" --no-sync python - <<'PY'
import importlib.metadata as metadata
import sys
import time

from physicalai_gr00t_policy_plugin import environments
import physicalai_gr00t_policy_plugin.runtime  # noqa: F401

try:
    metadata.version("physicalai-train")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("physicalai-train must not be installed")
assert not any(name == "physicalai.engine" or name.startswith("physicalai.engine.") for name in sys.modules)

for environment in (environments.NATIVE, environments.OPENVINO):
    started = time.monotonic()
    python, project, managed = environments.resolve(environment)
    assert managed and python.is_file()
    print(f"{environment.name}: prepared in {time.monotonic() - started:.1f}s; key={environment.key()}")
    print(f"  python={python}; project={project}; gr00t={environments.locked_version(environment, project)}")
    assert not any(name == "physicalai.engine" or name.startswith("physicalai.engine.") for name in sys.modules)
PY
```

For a real Runtime inference check, first run B3 below to create native/OpenVINO packages and the pinned LIBERO observation fixture. Export three observations for the Runtime-only process (run from `library/`):

```bash
results_dir=/tmp/isolate-validation-gr00t-e2e
uv run --no-sync python - "$results_dir/data/libero_demo" /tmp/gr00t-observations.npz <<'PY'
from pathlib import Path
import sys
import numpy as np
from tests.integration.test_engine_gr00t_e2e import _observations

rows = _observations(Path(sys.argv[1]))
np.savez_compressed(
    sys.argv[2],
    images=np.stack([row["images"]["image"] for row in rows]),
    wrist_images=np.stack([row["images"]["wrist_image"] for row in rows]),
    states=np.stack([row["state"] for row in rows]),
    tasks=np.asarray([row["task"] for row in rows]),
)
PY
```

Run `InferenceModel` in the separate environment (substitute the same `results_dir` and NPZ paths):

```bash
uv run --project "$runtime_env" --no-sync python - "$results_dir" /tmp/gr00t-observations.npz <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path
import numpy as np
from physicalai.inference import InferenceModel

try:
    metadata.version("physicalai-train")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("physicalai-train must not be installed")

with np.load(sys.argv[2], allow_pickle=False) as data:
    rows = [
        {
            "images": {"image": data["images"][i], "wrist_image": data["wrist_images"][i]},
            "state": data["states"][i],
            "task": str(data["tasks"][i]),
        }
        for i in range(len(data["states"]))
    ]

packages = Path(sys.argv[1]) / "released" / "packages"
actions = {}
for backend in ("native", "openvino"):
    with InferenceModel(packages / backend, device="auto", seed=1234) as model:
        actions[backend] = np.stack([model.predict_action_chunk(row) for row in rows])
        model.reset()
        replay = model.predict_action_chunk(rows[0])
    np.testing.assert_array_equal(actions[backend][0], replay)
    assert actions[backend].shape == (3, 16, 7)
    assert np.isfinite(actions[backend]).all()

assert not any(name == "physicalai.engine" or name.startswith("physicalai.engine.") for name in sys.modules)
difference = np.abs(actions["native"] - actions["openvino"])
mean_abs = float(difference.mean())
limit = 0.05 * max(float(np.abs(actions["native"]).mean()), 1e-3)
assert mean_abs < limit
print(f"Runtime-only inference pass: shape={actions['native'].shape}, max_abs={difference.max():.6g}, mean_abs={mean_abs:.6g}, limit={limit:.6g}")
PY
```

`uv` may warn that the host `VIRTUAL_ENV` differs from an isolate-managed worker environment. It ignores that variable and uses the declared Runtime profile; the warning is not a failure.

### B — LeRobot training and inference parity (CUDA)

From `library/`, run each opt-in integration suite separately:

```bash
PHYSICALAI_RUN_LEROBOT_INTEGRATION=1 PHYSICALAI_LEROBOT_TEST_STEPS=5 PHYSICALAI_LEROBOT_DEVICE=cuda \
  uv run --no-sync pytest tests/integration/test_engine_lerobot_hf_training.py -q --tb=short

PHYSICALAI_RUN_LEROBOT_PRETRAINED_INFERENCE=1 PHYSICALAI_LEROBOT_DEVICE=cuda \
  uv run --no-sync pytest tests/integration/test_engine_lerobot_pretrained_inference_parity.py -q --tb=short
```

The first command runs the small pinned-dataset training cases and native-versus-Engine ACT parity. The second downloads or uses cached immutable revisions of pretrained ACT/Diffusion policies and their datasets, then compares three actions plus reset/replay. `PHYSICALAI_LEROBOT_DEVICE=cuda` is explicit so a GPU-equipped host cannot silently run the opt-in parity on CPU.

### B3 — GR00T native and OpenVINO end-to-end

This test downloads/uses the pinned LIBERO demo and GR00T checkpoint; it needs CUDA and Hugging Face access to the backbone. It runs a 20-step training/export workflow and a released-checkpoint workflow by default.

```bash
PHYSICALAI_RUN_GR00T_E2E=1 PHYSICALAI_GR00T_E2E_OUTPUT=/tmp/isolate-validation-gr00t-e2e \
  uv run --no-sync pytest tests/integration/test_engine_gr00t_e2e.py -s --tb=short
```

The test writes `report.json` under `PHYSICALAI_GR00T_E2E_OUTPUT`. It prepares the native and OpenVINO environments, packages both formats, checks reset/replay and backend parity, and compares the released checkpoint's actions against the dataset and zero-action baseline.

### C — GR00T plugin tests

From `library/`, run the plugin tests (including the isolate-only import guard, environment declarations, and published-patch enablement metadata):

```bash
uv run --no-sync pytest ../plugins/policies/gr00t/tests -q --tb=short
```

## Results on this host

All durations below are pytest-reported durations unless noted. The final runs used the SHAs and environment above.

| Check                                       | Result            | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Runtime-only import and prepare (A1)        | **Pass**          | In a separate environment, `physicalai-train` was absent; importing the GR00T Runtime modules left `physicalai.engine` unloaded. `environments.resolve()` prepared native in 10.9 s and OpenVINO in 18.6 s; both reported locked GR00T `0.1.0`. Keys: native `b10b00080b68c018ec08e41739ab7e413adfe34ecc33c216d0cf5840dd661444`, OpenVINO `e9219a77efc5651dd6189b9bab34c7bd4dc929f486d219e6877bd611d5b10c3d`.                                                                                                                  |
| Runtime-only native/OpenVINO inference (A1) | **Pass**          | `physicalai.inference.InferenceModel` loaded both released packages with `physicalai-train` absent and `physicalai.engine` still unloaded. Each produced finite `(3, 16, 7)` actions and exact reset replay. Native/OpenVINO max absolute difference `0.00629660`, mean absolute difference `0.00097810896`, below the `0.0112431` mean-difference limit.                                                                                                                                                                      |
| LeRobot training and native parity (B)      | **Pass**          | `test_engine_lerobot_hf_training.py`: **13 passed** in 63.28 s; three five-step CUDA training/package cases and one native-versus-Engine parity case completed. The parity run had zero loss difference, zero maximum weight difference, and zero mismatched weights (`rtol=1e-6`, `atol=1e-7`).                                                                                                                                                                                                                               |
| Pretrained LeRobot inference parity (B)     | **Pass**          | `test_engine_lerobot_pretrained_inference_parity.py`: **2 passed** in 13.81 s on CUDA. ACT/Aloha and Diffusion/PushT each ran three steps; both reported max absolute native-versus-Runtime action difference `0.0` (`rtol=1e-3`, `atol=1e-4`) and passed reset/replay.                                                                                                                                                                                                                                                        |
| GR00T end-to-end / OpenVINO (B3)            | **Pass**          | `test_engine_gr00t_e2e.py`: **2 passed** in 536.25 s (8m56s). The 20-step, batch-8 trained package had exporter cosine `0.9999999404`; native/OpenVINO actions had max absolute difference `0.00498119`, mean `0.000452425`, and exact reset replay. The released checkpoint had exporter cosine `0.999999404`; native/OpenVINO max absolute difference `0.00629660`, mean `0.000978109`; native and OpenVINO chunk MAE were `0.0132632` and `0.0134482`, versus `0.215913` for zero actions. OpenVINO inference selected CPU. |
| GR00T plugin tests (C)                      | **Pass**          | `plugins/policies/gr00t/tests`: **50 passed** in 4.43 s.                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Worker protocol regression                  | **Pass**          | `tests/unit/engine/test_runner.py`: **7 passed** in 0.42 s, including a `run_worker()` test with a carriage-return progress update before a tagged progress event.                                                                                                                                                                                                                                                                                                                                                             |
| Pyrefly type check                          | **Failed**        | Reported two existing `unbound-name` diagnostics for `identity` and `key` at `library/src/physicalai/engine/_environment.py:466`; that file is unchanged. Mypy on the changed runner passed.                                                                                                                                                                                                                                                                                                                                   |
| Full library test suite                     | **Not completed** | Started once, then stopped at the user's request after the partial output showed failures/errors in the optional-dependency suite. The run produced no final count and is out of scope for this validation PR.                                                                                                                                                                                                                                                                                                                 |

The first opt-in training attempt exposed a progress-reporting defect: worker events were recognized only when the protocol marker began a stderr line, but `tqdm` can write a carriage-return update before that marker. Training completed, but the LeRobot hook missed a step and the GR00T hook received no steps. The minimal fix scans each line for the tagged record while preserving preceding log text; the regression test exercises the public `run_worker()` boundary. After the fix, the LeRobot and GR00T suites above passed. No Engine or isolate API changed.

The Runtime-only inference check used the released checkpoint/package and observations from the pinned LIBERO fixture. The GR00T end-to-end test also produced native and OpenVINO exports from a 20-step fine-tune. This evidence covers the tested revisions and host only; it does not replace validation on other drivers, hardware, or framework revisions.
