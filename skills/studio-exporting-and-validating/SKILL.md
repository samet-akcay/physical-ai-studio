---
name: studio-exporting-and-validating
description: Export and validate Physical AI Studio policies for Runtime deployment. Use when working on policy.export(...), physicalai export, OpenVINO, ONNX, Torch, ExecuTorch, export metadata, numerical parity, backend-specific export failures, or the Studio side of the export/load contract consumed by Runtime InferenceModel.load(...).
license: Apache-2.0
---

# Exporting and Validating Studio Policies

Use this skill for changes involving `library/src/physicalai/export`, policy `export(...)` support, or the `physicalai export` CLI.

## Workflow

1. Identify the source policy, checkpoint/config path, target backend, and expected Runtime loader behavior.
2. Confirm whether export is through Python (`policy.export(...)`) or CLI (`physicalai export ...`). Both routes must preserve the same artifact contract.
3. Inspect backend-specific constraints before changing generic export code.
4. Validate numerical parity against the Torch policy path for representative inputs when possible.
5. Validate artifact structure and metadata against `references/export-contract.md`.
6. If the exported artifact is intended for Runtime, confirm Runtime can auto-detect or explicitly load the backend with `InferenceModel.load(...)`.

## Backend Notes

- OpenVINO and ONNX are deployment-oriented backends consumed by the Runtime package.
- Torch export is useful for development/debugging and may require Runtime support from a companion adapter.
- ExecuTorch is optional and dependency-sensitive. Treat it as available only when the required optional packages or companion distribution are installed.

## Required Checks

- The export path contains the expected model file and metadata files.
- Metadata names inputs/outputs/features consistently with Runtime preprocessing.
- Backend-specific dependencies are imported lazily or guarded with helpful errors.
- CLI docs and Python API examples stay consistent.

## References

- See `references/export-contract.md` for artifact requirements shared with Runtime.
- See `references/openvino.md`, `references/onnx.md`, `references/torch.md`, and `references/executorch.md` for backend-specific constraints.
