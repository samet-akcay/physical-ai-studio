# Policy Evaluation Design

This document defines what part of benchmarking belongs in `physicalai` and what stays in `physicalai-train`.

The main split:

```text
InferenceModel     computes actions
PolicyRuntime      runs a robot control loop
PolicyEvaluator    evaluates a policy over episodes/tasks
```

## 1. Goal

Support this in `physicalai` without torch:

```python
from physicalai.benchmark import EvaluationConfig, PolicyEvaluator
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")

evaluator = PolicyEvaluator(
    envs=[my_env],
    policy=model,
    config=EvaluationConfig(num_episodes=20, max_steps=300),
)

results = evaluator.evaluate()
print(results.summary())
print(results.metadata["latency"])
```

## 2. What Belongs Where

| Thing | `physicalai` | `physicalai-train` |
| --- | --- | --- |
| `Observation` runtime contract | yes | uses it |
| exported model evaluation | yes | uses it |
| latency / FPS metrics | yes | uses it |
| benchmark result objects | yes | uses it |
| torch policy support | no | yes, via adapters |
| LIBERO / train gym presets | no | yes |
| production robot loop | no | no, belongs in `PolicyRuntime` |

## 3. Objective Stance

### Observation

`Observation` should be in `physicalai` if it is the canonical runtime contract:

```text
robot -> observation -> preprocess -> model -> action
```

Constraints:

- numpy / python only
- minimal runtime fields
- no torch dependency

### Benchmark

Not all of benchmark belongs in `physicalai`.

Only the runtime-safe core belongs there:

```text
env episodes + policy calls + metrics + latency
```

Train-specific presets and torch integration stay in `physicalai-train`.

## 4. Three Different Concerns

```text
task evaluation      does the policy solve the task?
model evaluation     how fast is select_action()?
runtime evaluation   how well does PolicyRuntime behave?
```

The first two belong in the evaluation API.

The third is separate and should not be mixed into the normal episode evaluator.

## 5. Naming

Avoid `BenchmarkRunner` because `InferenceRunner` already exists.

Use:

```text
InferenceRunner   model execution strategy
PolicyEvaluator   episode/task evaluation orchestration
PolicyRuntime     production control loop
```

## 6. Runtime Evaluation Protocols

`physicalai` should expose small numpy-only protocols:

```python
class PolicyLike(Protocol):
    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...
    def reset(self) -> None: ...


class EnvLike(Protocol):
    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict]: ...
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]: ...
    def close(self) -> None: ...
```

Why protocols exist:

- exported `InferenceModel` can satisfy `PolicyLike`
- custom envs can satisfy `EnvLike`
- no torch at import time
- no dependency on training gym classes

## 7. Exported Model Performance

Exported model performance is measured by timing `select_action()` during rollout:

```python
while not done:
    t0 = time.perf_counter()
    action = policy.select_action(obs)
    latency_s = time.perf_counter() - t0

    obs, reward, terminated, truncated, info = env.step(action)
```

This gives two categories of metrics:

```text
task metrics     success rate, reward, episode length
latency metrics  mean, p50, p95, p99, max
```

## 8. `InferenceModel` Compatibility

The runtime-facing contract should be numpy-native:

```python
class InferenceModel:
    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...
    def reset(self) -> None: ...
```

Then exported models work directly with `PolicyEvaluator`:

```python
from physicalai.benchmark import PolicyEvaluator
from physicalai.inference import InferenceModel

model = InferenceModel.load("./exports/act_policy")
results = PolicyEvaluator(envs=[env], policy=model).evaluate()
```

## 9. Train-Side Adapters

Unlike `Observation`, benchmark should not use extension hooks as the primary mechanism.

Use adapters instead:

```python
class TorchPolicyAdapter:
    def __init__(self, policy, device="cpu") -> None: ...
    def reset(self) -> None: ...
    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray: ...


class GymEnvAdapter:
    def __init__(self, gym) -> None: ...
    def reset(self, *, seed=None) -> tuple[dict[str, np.ndarray], dict]: ...
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]: ...
    def close(self) -> None: ...
```

Why adapters instead of `to.tensor()`-style extension:

- benchmark is orchestrating behavior, not converting one object
- both env and policy interfaces need adaptation
- wrappers keep the runtime import graph clean

## 10. Example Split

### `physicalai`

```text
physicalai/
  benchmark/
    __init__.py
    _protocols.py
    _evaluator.py
    _results.py
    _latency.py
```

### `physicalai-train`

```text
physicalai/
  benchmark/
    adapters.py
    libero.py
    pusht.py
```

## 11. What `PolicyEvaluator` Does Not Own

It should not become a second runtime:

- robot connection lifecycle
- action queue behavior
- async execution scheduling
- remote inference transport
- production control-loop timing

Those belong in `PolicyRuntime`.

## 12. Summary

```text
Observation in physicalai: yes
Benchmark core in physicalai: yes
Full train benchmark stack in physicalai: no

Use converters for Observation.
Use adapters for benchmark.
Keep runtime evaluation separate from PolicyRuntime benchmarking.
```
