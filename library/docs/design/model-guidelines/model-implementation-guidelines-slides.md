---
marp: true
theme: default
paginate: true
transition: fade
style: |
  section {
    font-size: 28px;
  }
  table {
    font-size: 22px;
  }
  h1, h2 {
    color: #0068b5;
  }
  section.lead h1 {
    font-size: 56px;
  }
---

<!-- _class: lead -->

# Model Implementation Guidelines

## Studio-Enabled First

Model coverage · XPU execution · Runtime export

---
<!-- transition: fade -->

# TLDR

Increase model coverage without making OpenVINO export block initial support.

* Ship `Studio-enabled` support first when XPU capabilities are validated.
* Continue Runtime/OpenVINO export, parity, and quantization work toward full completion.
* Treat OpenVINO export as expected for full deployment readiness when feasible.
* Use wrappers first; attempt wrapper export before first-party code.
* Build first-party only when wrappers cannot meet required capabilities.

---
<!-- transition: fade -->

# Why Change

Model support is too slow when every policy waits for full export support.

* Users need broad model coverage.
* XPU training or inference is useful before export is ready.
* New policies often ship with CUDA-only dependencies.
* First-party ports should be reserved for cases that need them.

---
<!-- transition: fade -->

# Completion States

Use two states so support can ship without overstating completion.

| State | Meaning |
|---|---|
| `Studio-enabled` | Usable in Studio for documented capabilities |
| `Fully complete` | Required Studio and Runtime capabilities validated |

`Studio-enabled` ships first. Full completion stays open until expected Runtime/OpenVINO work is done or marked not applicable.

---
<!-- transition: fade -->

# Implementation Order

Use the lowest-cost implementation that meets the required capabilities.

| Option | Use when | Cost |
|---|---|---|
| Generic wrapper | Upstream behavior is enough | Hours |
| Named wrapper | Studio needs policy-specific support | Hours-days |
| First-party | Wrapper cannot meet requirements | Weeks |

Do not build first-party only to unblock initial XPU support.

---
<!-- transition: fade -->

# Enablement Flow

```text
Initial evaluation
        ↓
Reference baseline
        ↓
Validate XPU training or Torch/XPU inference
        ↓
Ship Studio-enabled support
        ↓
Try wrapper export when Runtime/export value exists
        ↓
Fully complete, or keep completion item open
```

---
<!-- transition: fade -->

# Export Rule

OpenVINO export is expected for full deployment readiness when technically feasible.

* It should not block initial Studio enablement.
* Try export through the wrapper first.
* Use direct PyTorch-to-OpenVINO when supported.
* Use ONNX as interchange when needed.
* Validate parity, Runtime metadata, and quantization when required.

---
<!-- transition: fade -->

# Upstream Rule

Offer generic fixes upstream, but do not wait for upstream review to ship support.

* Ship required downstream fixes while review is pending.
* Link downstream patches to upstream issues or PRs.
* Delete downstream patches after upstream support lands.

---
<!-- transition: fade -->

# Cross-Team Work

| Work area | Covers | Owner |
|---|---|---|
| Platform | PyTorch XPU ops, kernels, CUDA-only alternatives | Platform teams |
| Upstream | LeRobot CI, export contract | Studio + partnership or DevRel |
| Product | Studio/Runtime, OpenVINO export, quantization | Studio team |

Studio provides a blocker list grouped by root cause.

---
<!-- transition: fade -->

# Requested Decisions

| Decision | Needed from |
|---|---|
| LeRobot / Hugging Face owner | Partnership or DevRel |
| CI hardware allocation | Infrastructure owner |
| Triage cadence | PyTorch XPU and OpenVINO leads |
| Robot-learning workload domain | Platform roadmap owners |

These decisions do not block Studio work. They reduce support cost over time.

---
<!-- transition: fade -->

# Summary

* Ship `Studio-enabled` support first.
* Keep Runtime/OpenVINO work open until full completion.
* Use wrappers before first-party implementations.
* Track required capabilities, supported capabilities, state, and owners.

Main doc: `model-implementation-guidelines.md`
Reference: `model-implementation-guidelines-reference.md`
Cross-team: `intel-enablement-strategy.md`
