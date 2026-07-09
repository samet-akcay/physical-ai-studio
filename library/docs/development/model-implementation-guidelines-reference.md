# Model Implementation Guidelines Reference

**Status**: Draft for team review
**Scope**: Detailed reference material for [Model Implementation Guidelines](./model-implementation-guidelines.md).

---

## Initial Evaluation Scorecard

Before implementation, create a tracking issue and record this scorecard.

| Criterion | Question | Weight |
|---|---|---|
| Traction | Is adoption growing through stars, citations, downloads, or user reports? | High |
| Customer or product need | Is there a customer, demo, benchmark, or committed roadmap dependency? | High |
| License | Are code and weights compatible with Studio distribution? | Required |
| Architecture risk | Does the model introduce new op families or export patterns? | Medium |
| Dependency risk | Does it require CUDA-only packages, custom kernels, Triton, or unstable dependencies? | Medium |
| Host framework | Is it in LeRobot, another framework, or a standalone repo? | Info |
| Reuse | Does support unlock a family of related policies? | Medium |

Decision outcomes:

- `reject`: do not invest; record the reason.
- `track`: monitor and revisit when evidence changes.
- `generic wrapper`: allow use through a generic wrapper only; no policy-specific Studio support.
- `named wrapper`: add a named Studio wrapper and validate the required XPU capabilities.
- `first-party implementation`: build and maintain a native Studio policy.

For `named wrapper` and `first-party implementation`, record the required capability set. Full completion is measured against that record.

Example: `XPU fine-tuning and OpenVINO export required; ExecuTorch not required`.

## Wrapper-First Export Details

Export is a Runtime and deployment optimization path, not a prerequisite for all model support.

For a named wrapper, attempt export through the wrapped policy before starting a first-party implementation when Runtime deployment is needed or export provides value.

The export path should use the same Studio contract expected by Runtime:

- OpenVINO artifacts for Intel deployment.
- ONNX artifacts when needed as an interchange format or deployment target.
- ExecuTorch artifacts when required, including OpenVINO-backed ExecuTorch flows where applicable.
- Export metadata required by Runtime.
- Numerical parity against the reference baseline.
- Quantization when the exported graph is stable.
- `InferenceModel.load(...)` validation when Runtime support is in scope.

Generic export fixes should be offered upstream. Examples include exportable policy cores, removal of data-dependent Python control flow, dynamic-shape declarations, and separation of model state from host-side action selection.

## Enablement Pipeline Details

### Phase 1: Reproduce the Reference

Run the model in its native environment before modifying it.

Record:

- Upstream commit, package versions, checkpoint, seeds, and hardware.
- Forward-pass outputs on fixed inputs.
- Fine-tuning curve on a small standard dataset or gym.
- Benchmark success rate.

The baseline is the comparison point for XPU enablement, wrapper export, first-party ports, and quantization.

### Phase 2a: Enable XPU Training and Inference

Classify failures as follows.

| Class | Examples | Action |
|---|---|---|
| Device assumptions | Hardcoded `"cuda"`, `torch.cuda.*`, missing device plumbing | Fix locally and upstream |
| CUDA-only dependency | `flash-attn`, custom CUDA kernels | Replace behind a capability check without changing model math |
| Missing op or poor kernel | Unsupported XPU op, severe perf issue | File a minimal repro with the PyTorch XPU team; carry a fallback if needed |

Validate training or fine-tuning against the Phase 1 baseline:

- Loss curve within the agreed tolerance.
- Benchmark success within the agreed budget.
- Throughput measured and reported.

When the supported capability is XPU inference only, validate:

- Forward-pass outputs match the Phase 1 baseline within tolerance.
- `select_action` behavior matches upstream on fixed inputs.
- Throughput is measured and reported.

### Phase 2b: Enable Export

Attempt export through the named wrapper first when Runtime deployment is needed or export provides value.

For OpenVINO, use direct PyTorch export when supported. Use ONNX as an interchange path when direct export is unavailable or less reliable.

Common blockers:

| Blocker | Preferred fix |
|---|---|
| Data-dependent Python control flow | Traceable tensor ops or `torch.cond` |
| Dynamic shapes | Explicit dynamic dimensions |
| Host-side state in `select_action` | Exportable model core plus host-managed state |
| Stochastic heads | Export deterministic core; sample on host |

Then:

1. Produce the required OpenVINO, ONNX, and ExecuTorch artifacts for the deployment target.
2. Check numerical parity against the Phase 1 outputs with backend-specific tolerances.
3. Quantize with NNCF or another approved path.
4. Validate quantized artifacts on task-level metrics, not tensor error alone.
5. File OpenVINO or ExecuTorch gaps with minimal exported graphs and carry decompositions only when needed.

## LeRobot Details

LeRobot is important to Studio model enablement. Many open policies land there first, and the Studio wrapper layer (`physicalai.policies.lerobot`) makes them usable in the Lightning pipeline without a port.

How Studio uses LeRobot:

1. **Use through wrappers.** Generic and named wrappers give access to LeRobot policies without porting.
2. **Wrapper export as a downstream capability.** LeRobot does not support model export today. Studio can add export on top of the wrappers.
3. **Upstream export as the preferred end state.** When LeRobot adds export support upstream, remove the downstream export shim and keep only the Studio contract layer.
4. **XPU fixes upstream first.** Offer fixes upstream, ship downstream while review is pending, delete when merged.

LeRobot coordination is covered in [Intel Hardware Enablement](./intel-enablement-strategy.md): export contract, Intel-hosted CI, and named contacts.

## Common Cases

### Policy Released in LeRobot

1. Run the initial evaluation; expose through the generic wrapper if it works.
2. Reproduce the upstream baseline; fix XPU issues in LeRobot where possible, with wrapper shims as needed.
3. Add a named wrapper and equivalence tests when policy-specific support is needed.
4. Ship Studio-enabled support even if export is blocked; continue Runtime/OpenVINO work until full completion unless marked not applicable.
5. Build a first-party implementation only if wrapper support cannot meet required capabilities.

### Policy Released in a Standalone Repo

1. Run the initial evaluation with extra attention to license and dependency risk.
2. If the author plans LeRobot integration, prefer helping that path instead of building a second integration.
3. Wrap the repo if it is stable and installable; consider first-party only if it is unmaintained, not packageable, or required by a committed roadmap item.
4. Upstream generic fixes to the author repo, PyTorch, OpenVINO, or ExecuTorch as appropriate.

### New Framework

Build a plugin adapter first, add per-policy wrappers through it, and reuse the same equivalence tests, export contract, and implementation rules.

## Escalation Paths

| Blocker | Engage | Bring | Downstream action |
|---|---|---|---|
| XPU op failure or kernel performance | PyTorch XPU team | Minimal repro, op list, model context | Fallback decomposition or narrow CPU fallback |
| OpenVINO conversion or op gap | OpenVINO team | Exported graph and repro | Downstream decomposition |
| ExecuTorch lowering gap | ExecuTorch channel | Exported program and target backend | Keep the OpenVINO path working |
| Stalled upstream PR | Maintainers or partnership contact | Working PR, tests, offer of CI | Carry downstream patch |

## Definition of Done

### Named Wrapper

- [ ] Initial evaluation and required capability set recorded.
- [ ] Reference baseline pinned.
- [ ] XPU parity demonstrated for the supported capabilities.
- [ ] Named wrapper, aliases, and equivalence tests added.
- [ ] Training config added under `library/configs`.
- [ ] Supported capabilities documented.
- [ ] Wrapper export attempted when Runtime deployment is required or export adds value.
- [ ] Runtime export gaps documented when wrapper export is incomplete.
- [ ] Completion state recorded: Studio-enabled or fully complete.
- [ ] Completion item with an owner created when required capabilities are open.
- [ ] Generic fixes linked to upstream PRs or issues.
- [ ] Downstream patches documented.
- [ ] User-facing docs updated.

### First-Party Implementation

- [ ] Named wrapper requirements complete when a wrapper exists.
- [ ] First-party config, model, policy, and preprocessor added.
- [ ] Policy factory and package exports registered.
- [ ] Upstream, wrapper, and first-party outputs agree within tolerance.
- [ ] Required OpenVINO, ONNX, and ExecuTorch exports pass parity when not already satisfied by wrapper export.
- [ ] Quantized artifact meets task-level accuracy budget.
- [ ] Runtime `InferenceModel.load(...)` round trip validated.
- [ ] Benchmark results recorded.

### Full Completion

- [ ] Required Runtime inference path validated.
- [ ] Required OpenVINO export path validated, either direct PyTorch-to-OpenVINO or through an interchange path such as ONNX.
- [ ] Runtime metadata validated.
- [ ] Numerical parity validated against the reference baseline.
- [ ] Quantization validated when required.
- [ ] Remaining unsupported capabilities explicitly marked not required for the product use case.

## Risks

| Risk | Mitigation |
|---|---|
| Model support is delayed by export gaps | Ship Studio-enabled support first; track Runtime/OpenVINO completion separately |
| Studio-enabled becomes the permanent state | Completion item with an owner per open required capability; quarterly review of open items |
| Upstream does not merge XPU or export fixes | Ship downstream patches; keep upstream PRs open and linked |
| Partnership path is unavailable | Use normal upstream PRs and the downstream carry process |
| Upstream export makes a Studio shim redundant | Delete the shim and keep the Studio contract layer |
| First-party port drifts from upstream | Pin upstream versions and run equivalence CI |
| Enablement backlog exceeds capacity | Enforce initial evaluations and limit first-party implementations |
| Another framework replaces LeRobot | Add a plugin adapter and reuse the same implementation process |
| CUDA-only dependencies block XPU | Substitute supported implementations behind capability checks |
