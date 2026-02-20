# Physical-AI Packaging Strategy

Two repositories. Two PyPI distributions. One namespace.

**Repos:**

- [`physical-ai`](https://github.com/openvinotoolkit/physical-ai) — runtime/inference library (currently private)
- [`physical-ai-studio`](https://github.com/open-edge-platform/physical-ai-studio) — training SDK + application (going public)

**PyPI distributions:**

- `physicalai` — lightweight runtime (inference, capture, robot, export)
- `physicalai-train` — training SDK (policies, data, benchmarks, eval, gyms)

Both distributions share the `physicalai` namespace via PEP 420 implicit namespace packaging.

---

## Constraints

- Only **two repos** available (no third repo option)
- **No circular dependencies** — dependency flows upward only
- **Lightweight base install** — `pip install physicalai` must not pull torch/lightning
- **No `__init__.py` at namespace root** — PEP 420 requirement for namespace split
- Core APIs return plain `dict`/`np.ndarray` only
- Studio is **not** a PyPI package — built via npm or Docker only
- Training SDK **is** published to PyPI as `physicalai-train`

---

## Phased Rollout

### Phase 1 — Open-Source Launch

Timeline: immediate (3 days).

`physical-ai-studio` goes public with the codebase renamed from `getiaction` to `physicalai`. All Python modules ship inside a single distribution: `physicalai-train`.

```
physical-ai-studio/                    # goes public
├── application/                       # FastAPI + React (not packaged)
└── library/
    ├── pyproject.toml                 # name = "physicalai-train"
    └── src/physicalai/
        ├── train/                     # Trainer, Lightning integration
        ├── policies/                  # ACT, Pi0, SmolVLA, GR00T
        ├── data/                      # DataModule, Observation
        ├── benchmark/                 # LiberoBenchmark
        ├── eval/                      # Rollout evaluation
        ├── gyms/                      # Sim environments
        ├── inference/                 # InferenceModel (temporary)
        ├── export/                    # ONNX, OpenVINO export
        ├── transforms/                # Shared transforms
        ├── cli/                       # getiaction CLI
        ├── config/                    # Configuration
        └── devices/                   # Device utilities
        # NO physicalai/__init__.py

physical-ai/                           # stays private/empty
```

**Install experience:**

```bash
pip install physicalai-train           # gets everything
```

```python
from physicalai.train import Trainer
from physicalai.policies import ACT
from physicalai.inference import InferenceModel
```

**Why this works:** Shipping everything as `physicalai-train` avoids claiming the `physicalai` PyPI name prematurely. The `physicalai` name is reserved for the lightweight runtime that ships from `physical-ai` once that repo goes public. PEP 420 compliance (no namespace `__init__.py`) is enforced from day one so that Phase 2 is a clean split, not a rewrite.

---

### Phase 2 — Runtime Extraction

Timeline: when `physical-ai` repo goes public.

Extract runtime modules from studio into the `physical-ai` repo. Publish `physicalai` (lightweight) from there. `physicalai-train` stays in studio and declares `physicalai>=0.1.0` as a dependency.

```
physical-ai/                           # now public
├── pyproject.toml                     # name = "physicalai"
└── src/physicalai/
    ├── inference/                     # InferenceModel
    ├── capture/                       # Camera interfaces
    ├── robot/                         # Robot ABC
    ├── export/                        # Model export
    └── transforms/                    # Shared transforms
    # NO physicalai/__init__.py

physical-ai-studio/                    # training stays here
├── application/                       # FastAPI + React
└── library/
    ├── pyproject.toml                 # name = "physicalai-train"
    │                                  # depends on: physicalai>=0.1.0
    └── src/physicalai/
        ├── train/
        ├── policies/
        ├── data/
        ├── benchmark/
        ├── eval/
        ├── gyms/
        ├── cli/
        └── config/
        # NO physicalai/__init__.py
```

**Install experience:**

```bash
pip install physicalai                 # runtime only — no torch
pip install physicalai-train           # auto-pulls physicalai
```

```python
# Runtime only
from physicalai.inference import InferenceModel
from physicalai.capture import Camera

# Training (requires physicalai-train)
from physicalai.train import Trainer
from physicalai.policies import ACT
```

**What moves:**

| Module        | From                 | To            |
| ------------- | -------------------- | ------------- |
| `inference/`  | `physical-ai-studio` | `physical-ai` |
| `capture/`    | `physical-ai-studio` | `physical-ai` |
| `robot/`      | `physical-ai-studio` | `physical-ai` |
| `export/`     | `physical-ai-studio` | `physical-ai` |
| `transforms/` | `physical-ai-studio` | `physical-ai` |

**Prerequisite:** Clean up cross-module imports. Currently `inference/` imports from `data/`, `export/`, and `policies/` — these dependencies must be broken before extraction.

---

### Phase 3 — Training Consolidation (Pending Stakeholder Alignment)

Timeline: requires executive approval.

Move training modules from studio into `physical-ai`. Both distributions publish from a single repo. Studio becomes a pure application (FastAPI + React).

```
physical-ai/                           # owns all Python code
├── packages/
│   ├── physicalai/                    # runtime dist
│   │   ├── pyproject.toml
│   │   └── src/physicalai/
│   │       ├── inference/
│   │       ├── capture/
│   │       ├── robot/
│   │       ├── export/
│   │       └── transforms/
│   │       # NO physicalai/__init__.py
│   └── physicalai-train/              # training dist
│       ├── pyproject.toml
│       └── src/physicalai/
│           ├── train/
│           ├── policies/
│           ├── data/
│           ├── benchmark/
│           ├── eval/
│           ├── gyms/
│           ├── cli/
│           └── config/
│           # NO physicalai/__init__.py

physical-ai-studio/                    # pure application
├── application/
│   ├── backend/                       # FastAPI
│   └── ui/                            # React
└── pyproject.toml                     # depends on physicalai-train
```

**Why this is the end-state:** Library code belongs in a library repo. Studio becomes a consumer, not a publisher. One repo for all Python packaging simplifies CI, versioning, and release coordination.

**Why it needs approval:** Moving training out of studio changes repo ownership boundaries. This is an organizational decision, not a technical one. If rejected, Phase 2 is a stable long-term state — training stays in studio, runtime lives in `physical-ai`, namespace split works either way.

---

## Package Architecture

### Namespace Split

Both distributions contribute subpackages under the `physicalai` namespace. Neither distribution owns the namespace root.

```
physicalai/                            # PEP 420 namespace — no __init__.py
├── inference/    → physicalai dist    # runtime
├── capture/      → physicalai dist
├── robot/        → physicalai dist
├── export/       → physicalai dist
├── transforms/   → physicalai dist
├── benchmark/    → physicalai dist    # evaluation mechanism (numpy-only)
├── train/        → physicalai-train   # training
├── policies/     → physicalai-train
├── data/         → physicalai-train
├── eval/         → physicalai-train   # rollout loop, video recording (torch)
├── gyms/         → physicalai-train   # simulation environments (heavy deps)
├── cli/          → physicalai-train
└── config/       → physicalai-train
```

Training subpackages (`train/`, `policies/`, `data/`, etc.) are siblings under `physicalai`, not nested under `physicalai.train.*`. This keeps imports flat and natural.

### Benchmark Split

Benchmarking spans both distributions. The **evaluation mechanism** belongs in the runtime dist. The **benchmark suites and simulation environments** belong in the training dist.

**`physicalai.benchmark`** (runtime dist, numpy-only):

- Environment protocol: `reset() → dict[str, np.ndarray]`, `step(action) → (obs, reward, done, info)`
- Policy protocol: `select_action(obs) → np.ndarray`, `reset()`
- `BenchmarkRunner` — the evaluation loop + timing + metrics aggregation
- `BenchmarkResults` — success rate, latency stats, episode data
- No torch. No gym. No simulation deps.

**Benchmark content in `physicalai-train`** (training dist, heavy deps):

- `LiberoBenchmark`, `PushTBenchmark` — specific benchmark presets that build on `physicalai.benchmark`
- Gym adapters wrapping simulation envs (LIBERO, PushT) into the runtime protocol
- Policy adapters wrapping Lightning modules into the runtime protocol
- Video recording, visualization

**Why this split:** The runtime dist owns `InferenceModel` — it should also own the ability to evaluate one. A deployer validating an exported model on edge hardware shouldn't need torch installed to measure success rate and latency. This mirrors OpenVINO (`benchmark_app` ships in the runtime, not in a training package) and Hugging Face (`evaluate` is a separate package from `transformers`).

### Dependency Direction

```
physicalai-train  →  physicalai  →  (numpy, opencv, etc.)
     │                    │
     │                    └── no torch, no lightning
     └── torch, lightning, policies, data
```

`physicalai-train` depends on `physicalai`. Never the reverse.

### Install Matrix

| User          | Install                        | Gets                                               |
| ------------- | ------------------------------ | -------------------------------------------------- |
| Edge deployer | `pip install physicalai`       | Inference, camera, robot, export, benchmark runner |
| ML researcher | `pip install physicalai-train` | Everything (auto-pulls physicalai)                 |
| Studio user   | Docker / npm build             | Application with physicalai-train baked in         |

### Import Examples

```python
# Runtime only (pip install physicalai)
from physicalai.inference import InferenceModel
from physicalai.capture import RealSenseCamera
from physicalai.robot import Robot
from physicalai.benchmark import BenchmarkRunner, BenchmarkResults

# Training (pip install physicalai-train)
from physicalai.train import Trainer
from physicalai.policies import ACT
from physicalai.data import LeRobotDataModule
from physicalai.gyms import LiberoGym
```

### Critical Rule

**No `__init__.py` at `src/physicalai/` in either distribution.** This is the PEP 420 namespace mechanism. If either package adds a `physicalai/__init__.py`, the namespace breaks — only one distribution's subpackages will be visible.

---

## PoC Validation

A proof-of-concept validates PEP 420 namespace packaging across both deployment scenarios. Location: `poc/` directory.

### Results: 22/22 passed

**Case 1 — Single repo, two distributions (11/11)**

Validates Phase 3 layout: both dists publish from one repo.

| Test                                                            | Result |
| --------------------------------------------------------------- | ------ |
| `pip install physicalai` → inference works                      | ✅     |
| `pip install physicalai` → `physicalai.train` does not exist    | ✅     |
| `pip install physicalai` → `physicalai.policies` does not exist | ✅     |
| `pip install physicalai-train` → auto-pulls `physicalai`        | ✅     |
| `physicalai.train` works after installing training dist         | ✅     |
| `physicalai.policies` works after installing training dist      | ✅     |
| Inference + train + policies work together in same process      | ✅     |
| Uninstall `physicalai-train` → inference survives               | ✅     |
| Uninstall `physicalai-train` → train is gone                    | ✅     |
| Editable install (`-e`) for both packages simultaneously        | ✅     |
| Editable: inference + train + policies all work together        | ✅     |

**Case 2 — Two repos, namespace split across repos (11/11)**

Validates Phase 2 layout: dists publish from separate repos.

| Test                                                                        | Result |
| --------------------------------------------------------------------------- | ------ |
| `pip install physicalai` (from physical-ai repo) → inference works          | ✅     |
| `pip install physicalai` → `physicalai.train` does not exist                | ✅     |
| `pip install physicalai` → `physicalai.policies` does not exist             | ✅     |
| `pip install physicalai-train` (from studio repo) → auto-pulls `physicalai` | ✅     |
| `physicalai.train` works after installing training dist                     | ✅     |
| `physicalai.policies` works after installing training dist                  | ✅     |
| Inference + train + policies work together (cross-repo)                     | ✅     |
| Uninstall `physicalai-train` → inference survives                           | ✅     |
| Uninstall `physicalai-train` → train is gone                                | ✅     |
| Editable install (`-e`) from two separate repos simultaneously              | ✅     |
| Editable: inference + train + policies all work together                    | ✅     |

### PoC Structure

```
poc/
├── validate.sh                            # runs both cases
├── case-1-single-repo/                    # Phase 3: two dists from one repo
│   ├── validate.sh
│   └── packages/
│       ├── physicalai/                    # runtime dist
│       │   ├── pyproject.toml
│       │   └── src/physicalai/inference/
│       └── physicalai-train/              # training dist
│           ├── pyproject.toml
│           └── src/physicalai/{train,policies}/
└── case-2-two-repos/                      # Phase 2: split across repos
    ├── validate.sh
    ├── physical-ai/                       # simulates physical-ai repo
    │   ├── pyproject.toml
    │   └── src/physicalai/inference/
    └── physical-ai-studio/                # simulates studio repo
        └── library/
            ├── pyproject.toml
            └── src/physicalai/{train,policies}/
```

### Reproduce

```bash
cd poc && bash validate.sh               # both cases
bash poc/case-1-single-repo/validate.sh  # Phase 3 layout
bash poc/case-2-two-repos/validate.sh    # Phase 2 layout
```

---

## Module Migration Map

Current `getiaction` modules → target distribution assignment.

| Current module | Target distribution | Target module        | Notes                                                                                                     |
| -------------- | ------------------- | -------------------- | --------------------------------------------------------------------------------------------------------- |
| `inference/`   | `physicalai`        | `inference/`         | Needs import cleanup first                                                                                |
| `devices/`     | `physicalai`        | `capture/`, `robot/` | Split into dedicated modules                                                                              |
| `export/`      | `physicalai`        | `export/`            | Remove torch module-level imports                                                                         |
| `transforms/`  | `physicalai`        | `transforms/`        | Shared — both dists may reference                                                                         |
| `benchmark/`   | Split across both   | `benchmark/`         | Runner + results + protocols → `physicalai`; LiberoBenchmark, PushTBenchmark presets → `physicalai-train` |
| `eval/`        | Split across both   | `eval/`              | Rollout protocol → `physicalai`; torch rollout, video recording → `physicalai-train`                      |
| `gyms/`        | `physicalai-train`  | `gyms/`              | Simulation environments (heavy deps)                                                                      |
| `train/`       | `physicalai-train`  | `train/`             | —                                                                                                         |
| `policies/`    | `physicalai-train`  | `policies/`          | —                                                                                                         |
| `data/`        | `physicalai-train`  | `data/`              | Remove torch module-level imports for runtime usage                                                       |
| `cli/`         | `physicalai-train`  | `cli/`               | —                                                                                                         |
| `config/`      | TBD                 | `config/`            | Depends on whether runtime needs it                                                                       |

### Import Cleanup Required Before Phase 2

- `inference/` imports from `data/`, `export/`, `policies/` — break these
- `export/mixin_export.py` imports `lightning`, `openvino`, `torch` at module level — defer or guard
- `data/observation.py` imports `torch` at module level — defer or guard
- `__init__.py` eagerly imports `Trainer` and XPU utilities — remove eager imports

---

## Risks

| Risk                                       | Mitigation                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------------- |
| PEP 420 unfamiliarity                      | PoC validates both scenarios; document the critical `__init__.py` rule            |
| Accidental `__init__.py` at namespace root | CI lint rule: fail if `src/physicalai/__init__.py` exists                         |
| Cross-distribution import leaks            | CI import boundary check: `physicalai` dist must not import from training modules |
| Runtime benchmark scope creep              | Keep `physicalai.benchmark` tiny (protocols + runner + results); no torch/gym     |
| Phase 3 blocked by stakeholders            | Phase 2 is a stable long-term fallback                                            |
| Version coordination between two dists     | Pin `physicalai>=X.Y.Z` in `physicalai-train`; release runtime first              |
| `transforms/` ownership ambiguity          | Assign to `physicalai`; training dist uses it as a dependency                     |

---

## Alternatives Considered

### Option A — Single Package With Extras

Ship everything as one distribution. Use extras for heavy dependencies (`pip install physicalai[train]`).

```python
pip install physicalai              # runtime only
pip install physicalai[train]       # adds torch, lightning
```

**Why rejected:**

- Requires strict lazy-import discipline across the entire codebase
- Risk of accidental heavy imports at runtime — one `import torch` at module level breaks lightweight install
- CI must test both base and extras installs
- Current codebase has module-level torch imports in `data/`, `export/`, `policies/` — would need extensive refactoring to make extras safe
- No hard dependency boundary — discipline is the only guard

### Option C — Training Under Studio Branding

Keep training SDK in `physical-ai-studio` repo published under a different package name (not `physicalai.*`).

```bash
pip install physicalai              # runtime
pip install physical-ai-studio-sdk  # training
```

**Why rejected:**

- Training lives under "studio" branding, which is confusing — studio is a UI application
- Breaks the `physicalai.*` namespace unity — users need to know two unrelated package names
- ML researchers searching PyPI for "physicalai" won't discover the training SDK
- Studio repo mixes UI application and library concerns permanently

---

_Last Updated: 2026-02-20_
