# Intel Hardware Enablement for Robot Learning

**Status**: Draft for team review
**Audience**: PyTorch XPU, oneAPI, OpenVINO, Studio, Runtime, and teams coordinating upstream work.
**Scope**: How Intel enables robot-learning workloads on Intel hardware across platform, upstream, and product work.

This document is the cross-team view. The per-model process is defined in [Model Implementation Guidelines](./model-implementation-guidelines.md).

---

## Executive Summary

Model support is slow when every policy is fixed only after release. Many new robot-learning policies use CUDA-only dependencies by default. Each release with these assumptions adds work for Intel support.

Intel enablement needs three coordinated work areas:

| Work area | Scope | Owner |
|---|---|---|
| Platform | PyTorch XPU ops, kernels, oneAPI, CUDA-only dependency replacements | PyTorch XPU and platform teams |
| Upstream | LeRobot integration, Intel CI, export contract, Hugging Face coordination | Studio with partnership or DevRel support |
| Product | Studio enablement, Runtime deployment, OpenVINO export, quantization | Studio team |

Studio enables models using the per-model process. During that work, Studio records missing XPU ops, kernel issues, CUDA-only dependencies, and export gaps. These gaps should become ranked input to the platform and upstream work areas.

The requested decisions are:

- Name an owner for LeRobot and Hugging Face coordination.
- Allocate CI hardware for LeRobot XPU and export smoke tests.
- Set a triage cadence for Studio-filed PyTorch XPU and OpenVINO gaps.
- Treat robot learning as a workload domain in platform planning.

These decisions do not block Studio work. Studio continues to ship supported capabilities first and tracks missing capabilities to completion.

---

## 1. Problem

Three facts drive the work:

1. New robot-learning policies are released faster than Intel can enable them one by one after release.
2. Many policies depend on CUDA-only packages such as `flash-attn`, custom CUDA kernels, or Triton kernels without XPU coverage.
3. LeRobot is where many open policies land first, and it is still practical to add Intel CI and export support upstream.

Per-model fixes are still required. They are not enough by themselves. The same blockers should be fixed once in the platform or upstream project when possible.

## 2. Work Areas

### 2.1 Platform

Platform work makes PyTorch XPU and oneAPI support the operators and kernels used by robot-learning policies.

Inputs from Studio:

- Minimal repros for missing or slow XPU ops.
- Affected models and workflows.
- Frequency: how many tracked models hit the same blocker.
- Measured impact after the fix lands.

Expected outputs:

- XPU op coverage for robot-learning workloads.
- Kernel performance fixes for common policy paths.
- Supported alternatives for CUDA-only dependencies, exposed behind capability checks.

### 2.2 Upstream

Upstream work reduces downstream patches by fixing shared projects directly.

Priority items:

1. **LeRobot CI on Intel hardware**: XPU training smoke tests and export smoke tests.
2. **LeRobot export contract**: an exportable policy core, host-managed state, and declared dynamic shapes.
3. **Named maintainer path**: contacts and triage for XPU and export issues.

Requirements:

- Working code.
- Minimal repros.
- CI capacity.
- Maintenance ownership for contributed changes.
- Vendor-specific behavior behind capability checks.

If LeRobot coordination does not happen, Studio continues with downstream patches and normal upstream PRs.

### 2.3 Product

Product work is controlled by the Studio team.

Scope:

- Studio-enabled support for XPU training, fine-tuning, or Torch/XPU inference.
- Runtime deployment through `InferenceModel.load(...)`.
- OpenVINO export when technically feasible and useful for deployment.
- Quantization when the exported graph is stable.

The product layer also validates platform and upstream fixes against complete workflows, not only isolated tests.

## 3. Feedback Loop

Studio produces the evidence used by the other work areas.

```text
Studio enables a model
        │
        ▼
Gaps found: XPU ops, kernel performance, CUDA-only deps, export blockers
        │
        ▼
Studio files minimal repros and ranks blockers by model impact
        │
        ├──► PyTorch XPU / oneAPI backlog
        ├──► OpenVINO / NNCF backlog
        └──► LeRobot fixes and export contract
        │
        ▼
Fix lands upstream; downstream patch is removed
```

Today, model status is tracked in a shared sheet. The missing view is a blocker list grouped by root cause. One CUDA-only dependency that blocks six models should be one ranked blocker with six-model impact, not six unrelated model rows.

Studio owns this blocker list.

Studio commits to:

- File each gap with a minimal repro.
- Include affected models, workflow, hardware, and measured impact.
- Deduplicate blockers across models.
- Re-rank blockers as the model list changes.
- Validate fixes on the affected workflows after they land.

## 4. Operating Rule

Use the same rule at every layer:

1. Ship the downstream fix when needed to unblock users.
2. Offer the generic fix upstream in parallel.
3. Link the downstream patch to the upstream issue or PR.
4. Delete the downstream patch after upstream support is available.

The goal is to avoid becoming a permanent porting function. Downstream patches are acceptable when they unblock users. They should not become the only path.

## 5. Requested Decisions

| Decision | Needed from | Purpose |
|---|---|---|
| LeRobot / Hugging Face owner | Partnership or DevRel | Establish maintainer contact and coordination path |
| CI hardware allocation | Infrastructure owner | Run LeRobot XPU and export smoke tests |
| Triage cadence | PyTorch XPU and OpenVINO leads | Review Studio-filed blockers as planning input |
| Robot-learning workload domain | Platform roadmap owners | Rank op and kernel work using robot-policy workloads |

None of these decisions block Studio. They determine whether support cost per model decreases over time.

## 6. Measures

Track outcomes, not activity counts.

| Measure | Target direction |
|---|---|
| Time from model release to Studio-enabled support | Down |
| Downstream patches carried per model | Down |
| Shared blockers affecting multiple models | Down |
| New LeRobot policies passing XPU smoke tests at release | Up |
| Upstream fixes merged relative to downstream patches added | Up |
| Models with validated OpenVINO deployment path | Up |

## 7. Risks

| Risk | Mitigation |
|---|---|
| LeRobot coordination does not happen | Continue normal upstream PRs and downstream patches |
| CUDA-only defaults become more common | Prioritize common blockers such as attention kernels |
| Platform teams cannot reserve triage time | Continue filing issues; record slower resolution as the impact |
| Studio blocker list is treated as unrelated support tickets | Report model count, workflow impact, and frequency for each blocker |
| Another framework replaces LeRobot | Apply the same CI, export, and adapter approach to that framework |
| Document stops matching practice | Review with the quarterly model enablement review |
