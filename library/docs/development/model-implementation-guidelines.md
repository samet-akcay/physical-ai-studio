# Model Implementation Guidelines

**Status**: Draft for team review
**Audience**: Studio team. This document defines the per-model process for Studio training and Runtime deployment.
**Scope**: How Studio evaluates, integrates, validates, exports, and maintains robot-learning policies for Intel hardware.

Reference details are in [Model Implementation Guidelines Reference](./model-implementation-guidelines-reference.md). Cross-team platform and upstream work is covered in [Intel Hardware Enablement for Robot Learning](./intel-enablement-strategy.md).

---

## Executive Summary

Studio should increase model coverage without making OpenVINO export block initial support.

- Ship `Studio-enabled` support first when XPU training, fine-tuning, or Torch/XPU inference is validated.
- Continue Runtime/OpenVINO export, parity, and quantization work toward full completion.
- Treat OpenVINO export as expected for full deployment readiness when technically feasible.
- Use wrappers first. Attempt wrapper export before first-party implementation.
- Build first-party implementations only when wrappers cannot meet the required capability set.
- Record each model's required capabilities, supported capabilities, completion state, and open completion items.

## 1. Core Policy

A policy can be supported for Studio use before all deployment work is done.

OpenVINO export is expected for full deployment readiness when technically feasible. It should improve inference performance, memory use, deployment portability, or Runtime integration. It should not block initial Studio enablement.

Capabilities can ship independently:

| Capability | Goal | Blocks Studio enablement? |
|---|---|---|
| XPU training and fine-tuning | Train or fine-tune on Intel GPUs | No |
| XPU inference | Run the policy on Intel GPUs without export | No |
| Runtime load | Load through the Runtime contract | No |
| OpenVINO export | Preferred Intel deployment path | No |
| Quantization | Improve deployment efficiency when export is stable | No |

Support claims apply only to validated hardware and workflows.

## 2. Completion States

Two completion states could be used.

| State | Meaning | Required evidence |
|---|---|---|
| `Studio-enabled` | The policy is usable in Studio for documented capabilities, such as XPU training or Torch/XPU inference | Reference baseline, capability validation, known gaps documented |
| `Fully complete` | The policy satisfies the required Studio and Runtime capabilities for the product use case | Studio validation plus required Runtime load, OpenVINO export, parity, and quantization evidence |

`Studio-enabled` support can ship before full completion. Full completion remains open while expected Runtime/OpenVINO export, parity, or quantization work is unfinished, unless that capability is explicitly marked not applicable for the product use case.

Each model must record:

- Required capability set.
- Supported capability set.
- Completion state.
- Open completion items and owners.

## 3. Implementation Options

Studio can use the lowest-cost implementation that meets the required capabilities.

| Option | Description | Typical cost | Support bar |
|---|---|---|---|
| Generic wrapper | Generic access path such as `LeRobotPolicy(policy_name=...)` | Hours | No guarantee beyond upstream behavior |
| Named wrapper | Named Studio class with aliases and equivalence tests | Hours to days | XPU capability validated; wrapper export attempted when Runtime deployment is needed or export provides value |
| First-party implementation | Native implementation under `physicalai/policies/<name>` | Weeks | XPU training, export, quantization, Runtime load, benchmarks |

We could use the following order:

1. Start with a wrapper when possible.
2. Add a named wrapper when policy-specific support is needed.
3. Attempt wrapper export before first-party implementation when Runtime deployment is needed or export provides value.
4. Build first-party only when wrapper support cannot meet required capabilities at acceptable cost.

We should not build a first-party implementation only to unblock initial XPU support.

## 4. Enablement Flow

1. **Initial evaluation**: decide whether to reject, track, wrap, or build first-party. Record the required capability set.
2. **Reference baseline**: run the upstream model first. Record versions, checkpoint, seeds, fixed inputs, metrics, and hardware.
3. **XPU enablement**: validate training, fine-tuning, or Torch/XPU inference against the baseline.
4. **Studio-enabled**: ship the validated Studio capability with supported hardware and known gaps documented.
5. **Wrapper export**: when Runtime deployment is needed or export provides value, attempt export through the wrapper first.
6. **Full completion**: close the model only when expected Runtime/OpenVINO export, parity, and quantization work is done or explicitly marked not applicable.

## 5. Contribution Policy

We should ideally offer generic fixes upstream, but should not wait for upstream review to ship user-facing support. Instead:

1. Offer generic fixes upstream: XPU device handling, CUDA-only dependency replacement, exportable control flow, dynamic-shape fixes, and removal of `.item()` in export paths.
2. Ship required fixes downstream while upstream review is pending.
3. Link each downstream patch to the upstream issue or PR.
4. Delete downstream patches after the upstream fix is available in the supported version.
5. Escalate stalled high-value PRs through maintainer or partnership channels when available. Do not depend on escalation for shipping.

## 6. API Ownership

Studio keeps its policy and export APIs even when upstream projects add similar features. (eg., lerobot adding export)

Reasons:

- Studio uses Lightning for training, configuration, checkpointing, logging, and distributed execution.
- Studio should support more than one upstream framework. (eg., starvla, openpi etc)
- Studio owns the export metadata, parity checks, quantization flow, and Runtime `InferenceModel.load(...)` contract.
- Studio must be able to patch CUDA-only dependencies, XPU gaps, and OpenVINO conversion issues without waiting on upstream release cycles.
- Studio must meet product requirements for licensing, dependency control, security review, and reproducibility.

When upstream support is sufficient, remove the downstream shim and keep only the Studio contract layer.

## 7. Review and Closure

`Studio-enabled` should not be a terminal state when required capabilities are still open.

Each model with open required capabilities keeps a completion item with an owner. The quarterly review lists `Studio-enabled` and `Fully complete` models. Each open item is either scheduled or its capability is removed from the required set with the reason recorded.

Main closure rules:

- A model can ship as `Studio-enabled` with export gaps documented.
- A model is `Fully complete` only when the required capability set is satisfied.
- OpenVINO export is expected for full deployment readiness when technically feasible.
- Unsupported capabilities must be explicitly marked not applicable for the product use case. (both in library and studio?)

## 8. References

- Detailed scorecards, export blockers, common cases, escalation paths, and checklists: [Model Implementation Guidelines Reference](./model-implementation-guidelines-reference.md)
- Cross-team platform and upstream work: [Intel Hardware Enablement for Robot Learning](./intel-enablement-strategy.md)
