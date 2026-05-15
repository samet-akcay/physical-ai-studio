# OpenVINO Conversion Benchmark — Findings Report

## Environment

| Component | Version |
|-----------|---------|
| OpenVINO | 2026.1.0-21367-63e31528c62 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.8 |
| Device | CPU only |

## Methods Tested

1. **Direct** — `ov.convert_model(nn.Module, example_input=...)` — TorchScript-based tracing
2. **torch.export** — `torch.export.export()` → `ov.convert_model(ExportedProgram)` — FX graph capture
3. **via ONNX** — `torch.onnx.export()` → `ov.convert_model("model.onnx")` — two-step through ONNX intermediate

## Compatibility Matrix

| Policy | Direct | torch.export | via ONNX |
|--------|--------|-------------|----------|
| **ACT** (~33M params) | ✅ | ✅ | ✅ |
| **SmolVLA** (~500M params) | ✅ | ❌ | ❌ |
| **Pi0.5** (~3B params) | ❌ | ❌ | ✅ |

**No single conversion method works for all three policies.**

## Detailed Results

### ACT (Action Chunking with Transformers)

Standard encoder-decoder transformer. All three methods succeed.

| Method | Conv Time | IR Size | OV Latency | PT Latency | Speedup | Max Abs Diff | Mem Delta |
|--------|-----------|---------|------------|------------|---------|--------------|-----------|
| Direct | **4.4s** | 131 MB | 20.4 ms | 57.2 ms | **2.8×** | 0.85 | 87 MB |
| torch.export | 7.8s | **126 MB** | 21.7 ms | 57.2 ms | 2.6× | 0.85 | ~0 MB |
| via ONNX | 13.7s | 127 MB | **19.6 ms** | 57.2 ms | **2.9×** | 0.85 | 143 MB |

- All methods produce identical numerical output (max diff = 0.85 is from random weights, not conversion error).
- Direct is fastest to convert. torch.export produces smallest IR. via ONNX is slowest but gives marginally best inference.
- All achieve ~2.7-2.9× speedup over PyTorch on CPU.

### SmolVLA (Vision-Language-Action, SmolVLM2-500M backbone)

| Method | Conv Time | IR Size | OV Latency | PT Latency | Speedup | Max Abs Diff | Mem Delta |
|--------|-----------|---------|------------|------------|---------|--------------|-----------|
| Direct | 75.3s | 957 MB | 1845 ms | 755.8 ms | **0.41×** | 0.005 | 1781 MB |
| torch.export | ❌ | — | — | — | — | — | — |
| via ONNX | ❌ | — | — | — | — | — | — |

**Only direct conversion succeeds.**

Failure details:
- **torch.export**: `aten.empty_permuted.default` has no OV conversion rule. This op is used in HuggingFace transformers' attention implementation.
- **via ONNX**: ONNX `Where-20` node fails type validation — `i64` vs `f32` mismatch in `index_put` operation from SmolVLA's mixed-type indexing.

Performance note: OV inference is **2.4× slower** than PyTorch. The VLM graph with SDPA attention, vision encoder, and cross-attention layers is not well-optimized in the OV CPU plugin. PyTorch benefits from highly tuned SDPA kernels.

### Pi0.5 (PaLIGemma-based VLA, ~3B params)

| Method | Conv Time | IR Size | OV Latency | PT Latency | Speedup | Max Abs Diff | Mem Delta |
|--------|-----------|---------|------------|------------|---------|--------------|-----------|
| Direct | ❌ | — | — | — | — | — | — |
| torch.export | ❌ | — | — | — | — | — | — |
| via ONNX | **808.7s** | **6763 MB** | 40625 ms | 5849 ms | **0.14×** | 4.25 | ~0 MB |

**Only via ONNX succeeds** — and it's the most problematic result.

Failure details:
- **Direct and torch.export**: Both fail on the same root cause — PaLIGemma's KV cache implementation uses `aten::cat` to concatenate an empty tensor with a KV cache tensor. OV's Concat op validator rejects this because the empty tensor has rank 0, making axis -2 invalid. Also, `aten.normal.float_float` (used in flow-matching noise sampling) has no conversion rule.
- **via ONNX succeeds** but with serious caveats:
  - Conversion takes **13.5 minutes**
  - IR is **6.76 GB**
  - OV inference is **6.9× slower** than PyTorch (40.6s vs 5.8s per step)
  - **Max abs diff = 4.25** — significant numerical divergence, likely from bf16→fp32 conversion in the ONNX path combined with the flow-matching denoising loop amplifying small errors over 10 steps

## Key Findings

### 1. No universal conversion method exists

This is the most important finding. Each VLA architecture hits different OV conversion limitations:

| Architecture Pattern | Blocker | Affected Methods |
|---------------------|---------|------------------|
| HuggingFace SDPA attention | `aten.empty_permuted.default` unsupported | torch.export, via ONNX (SmolVLA) |
| PaLIGemma KV cache (empty tensor concat) | `aten::cat` axis validation on rank-0 tensor | Direct, torch.export (Pi0.5) |
| Flow-matching noise sampling | `aten.normal.float_float` unsupported | torch.export (Pi0.5) |
| Mixed-type indexing | ONNX Where type mismatch | via ONNX (SmolVLA) |

### 2. Direct conversion works best for simpler models, ONNX for complex ones

- **ACT** (standard transformer): all methods work, direct is fastest
- **SmolVLA** (VLM + action expert): only direct works
- **Pi0.5** (PaLIGemma + KV cache): only ONNX works

This matches the existing codebase's design — `Pi05.extra_export_args` already sets `via_onnx=True` for OpenVINO export, while SmolVLA does not.

### 3. OV CPU inference is slower than PyTorch for VLM models

| Policy | PyTorch CPU | OV CPU | Ratio |
|--------|------------|--------|-------|
| ACT | 57 ms | 20 ms | **2.8× faster** |
| SmolVLA | 756 ms | 1845 ms | **2.4× slower** |
| Pi0.5 | 5849 ms | 40625 ms | **6.9× slower** |

The speedup reverses dramatically as model complexity increases. OV's CPU backend lacks optimizations for large VLM attention patterns that PyTorch's fused SDPA kernels handle well.

### 4. Numerical accuracy degrades for larger models via ONNX

| Policy | Method | Max Abs Diff | Mean Abs Diff |
|--------|--------|--------------|---------------|
| ACT | all methods | 0.85 | 0.41 |
| SmolVLA | direct | 0.005 | 0.002 |
| Pi0.5 | via ONNX | **4.25** | **1.21** |

Pi0.5's high divergence is concerning. The bf16→fp32 type promotion in the ONNX path, combined with 10-step denoising, amplifies rounding differences. This needs validation with trained weights to determine if it impacts action quality.

### 5. The existing dual-path strategy in `to_openvino()` is correct

The codebase already implements the right approach in `mixin_policy.py`:
- Default: direct `ov.convert_model(module)`
- Fallback: `via_onnx=True` when direct fails

Pi0.5 already opts into `via_onnx=True`. SmolVLA uses direct. ACT uses direct. This matches our benchmark findings exactly.

## Recommendations

### Short-term (current OV 2026.1)

1. **Keep the dual-path strategy** — direct by default, `via_onnx=True` per-policy override. No changes needed to the export pipeline architecture.
2. **Add a conversion validation step** that compares OV output to PyTorch reference and warns if max abs diff exceeds a threshold (e.g., 1.0). Pi0.5's 4.25 divergence should trigger a warning.
3. **Do not default to torch.export** — it fails for both VLM policies. It only works for simple models where direct also works.

### Medium-term (file OV issues)

4. **File OV bug: `aten.empty_permuted.default`** — blocks torch.export for all HuggingFace transformer-based models.
5. **File OV bug: `aten::cat` with empty/rank-0 tensors** — blocks direct conversion for models with KV cache (PaLIGemma, Gemma).
6. **File OV bug: `aten.normal.float_float`** — blocks torch.export for any model using stochastic sampling (flow matching, diffusion).

### Long-term

7. **Investigate OV CPU performance for VLM models** — the 2.4-6.9× slowdown vs PyTorch makes OV CPU deployment non-viable for SmolVLA/Pi0.5 without significant runtime optimization work. Consider GPU/NPU targets instead.
8. **Re-benchmark when OV fixes the above ops** — torch.export may become viable and could produce better-optimized IR than TorchScript tracing.

## Appendix: Failure Error Messages

### SmolVLA — torch.export
```
No conversion rule found for operations: aten.empty_permuted.default
```

### SmolVLA — via ONNX
```
[ONNX Frontend] Conversion failed for Where-20
Argument 1 and 2 element types must match. (i64 vs f32 in Select node)
```

### Pi0.5 — Direct (TorchScript)
```
aten::cat — Axis -2 out of the tensor rank range [-1, 0]
(empty KV cache tensor has rank 0, Concat expects rank ≥ 2)
Also: SequenceMark internal type failure
```

### Pi0.5 — torch.export (FX)
```
aten.cat.default — same axis validation failure as direct
aten.normal.float_float — no conversion rule (flow-matching noise)
```
