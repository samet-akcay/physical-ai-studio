# OpenVINO Conversion Benchmark — Findings Report

## Goal

Determine which OpenVINO conversion method works for each Vision-Language-Action (VLA) policy in Physical AI Studio, measure inference performance, and compare against PyTorch baselines on CPU and CUDA.

## Environment

| Component | Value |
|-----------|-------|
| OpenVINO | 2026.1.0 |
| PyTorch | 2.12.0+cu130 |
| Python | 3.12.3 |
| CPU | Intel Xeon W-2245 (16 threads) |
| GPU | NVIDIA RTX A6000 (48 GB) |
| OV inference device | CPU only (no Intel GPU on host; OV has no NVIDIA plugin) |
| PyTorch inference devices | CPU + CUDA |

## Methods Tested

| Method | Description |
|--------|-------------|
| **Direct** | `ov.convert_model(nn.Module, example_input=...)` — TorchScript-based tracing |
| **torch.export** | `torch.export.export()` then `ov.convert_model(ExportedProgram)` — FX graph |
| **via ONNX** | `torch.onnx.export()` then `ov.convert_model("model.onnx")` — two-step |

## Compatibility Matrix

| Policy | Direct | torch.export | via ONNX |
|--------|--------|--------------|----------|
| **ACT** (~33M params) | OK | OK | OK |
| **SmolVLA** (~500M params) | OK | FAIL | FAIL |
| **Pi0.5** (~3B params) | FAIL | FAIL | OK |

**No single conversion method works for all three policies.**

## Detailed Results

### ACT (Action Chunking with Transformers, ~33M params)

Standard encoder-decoder transformer for robotic manipulation.

PyTorch latency: **CPU 66.6 ms** · **CUDA 8.5 ms**

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | 5.7 | 131 | 28.8 | **2.3x faster** | 3.4x slower | 0.72 |
| torch.export | 8.6 | 126 | 33.5 | **2.0x faster** | 3.9x slower | 0.72 |
| via ONNX | 14.5 | 127 | 29.0 | **2.3x faster** | 3.4x slower | 0.72 |

All methods succeed and produce identical numerical output. OV-CPU is 2-2.3x faster than PyTorch-CPU, but 3.4x slower than PyTorch-CUDA. Direct is fastest to convert; torch.export produces the smallest IR; via ONNX is slowest to convert but marginally best at inference.

### SmolVLA (Vision-Language-Action, SmolVLM2-500M backbone, ~500M params)

VLM with SDPA attention, vision encoder, cross-attention, and action expert.

PyTorch latency: **CPU 898 ms** · **CUDA 235 ms**

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | 77.0 | 957 | 3287 | **3.7x slower** | **14x slower** | 0.004 |
| torch.export | FAIL | — | — | — | — | — |
| via ONNX | FAIL | — | — | — | — | — |

**Only direct conversion succeeds.** OV-CPU is 3.7x slower than PyTorch-CPU and 14x slower than PyTorch-CUDA. The VLM graph with SDPA attention is not well-optimised in the OV CPU plugin.

Failure details:
- **torch.export**: `aten.empty_permuted.default` has no OV conversion rule (used in HuggingFace transformers SDPA attention)
- **via ONNX**: `Where-20` node fails type validation — `i64` vs `f32` mismatch in `index_put` from mixed-type indexing

### Pi0.5 (PaLIGemma + flow-matching action expert, ~3B params)

Large VLA with PaLIGemma vision-language backbone, Gemma action expert, and 10-step flow-matching denoising.

PyTorch latency: **CPU 8464 ms** · **CUDA 252 ms** (33.6x CUDA speedup)

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | FAIL | — | — | — | — | — |
| torch.export | FAIL | — | — | — | — | — |
| via ONNX | 930 | 6763 | 77919 | **9.2x slower** | **309x slower** | **5.15** |

**Only via-ONNX succeeds**, and it is the most problematic result:
- Conversion takes **15.5 minutes**
- IR is **6.76 GB**
- OV-CPU inference is **309x slower** than PyTorch-CUDA (77.9 s vs 252 ms)
- **Max abs diff = 5.15** — significant numerical divergence from bf16-to-fp32 type promotion in the ONNX path, amplified over 10 denoising steps

Failure details:
- **direct**: PaLIGemma's KV cache uses `aten::cat` to concatenate an empty (rank-0) tensor with a KV cache tensor. OV's Concat validator rejects this (Axis -2 invalid for rank 0). Also hits `SequenceMark` internal type failure.
- **torch.export**: Same `aten.cat.default` axis issue, plus `aten.normal.float_float` (flow-matching noise sampling) has no conversion rule.

## Key Findings

### 1. No universal conversion method exists

Each architecture hits different OV conversion blockers:

| Architecture Pattern | Blocker | Affected Methods |
|---------------------|---------|------------------|
| HuggingFace SDPA attention | `aten.empty_permuted.default` unsupported | torch.export, via ONNX (SmolVLA) |
| PaLIGemma KV cache (empty tensor concat) | `aten::cat` axis validation on rank-0 tensor | direct, torch.export (Pi0.5) |
| Flow-matching noise sampling | `aten.normal.float_float` unsupported | torch.export (Pi0.5) |
| Mixed-type indexing | ONNX Where-20 type mismatch | via ONNX (SmolVLA) |

### 2. OV-CPU is only faster than PyTorch-CPU for simple transformers

| Policy | PT-CPU | PT-CUDA | OV-CPU (best) | OV vs PT-CPU | OV vs PT-CUDA |
|--------|-------:|--------:|--------------:|-------------:|--------------:|
| ACT | 66.6 ms | 8.5 ms | 28.8 ms | **2.3x faster** | 3.4x slower |
| SmolVLA | 898 ms | 235 ms | 3287 ms | 3.7x slower | 14x slower |
| Pi0.5 | 8464 ms | 252 ms | 77919 ms | 9.2x slower | 309x slower |

The OV-CPU advantage disappears for VLM models and reverses dramatically. PyTorch benefits from highly-tuned SDPA kernels on both CPU and CUDA that OV's CPU backend cannot match.

### 3. Numerical accuracy degrades for larger models via ONNX

| Policy | Method | Max Abs Diff |
|--------|--------|-------------:|
| ACT | all methods | 0.72 |
| SmolVLA | direct | 0.004 |
| Pi0.5 | via ONNX | **5.15** |

Pi0.5's high divergence is concerning. This needs validation with trained weights to determine if it impacts action quality in deployment.

### 4. The existing dual-path strategy in `to_openvino()` is correct

The codebase already implements the right approach:
- ACT, SmolVLA -> direct `ov.convert_model(module)` (default)
- Pi0.5 -> `via_onnx=True` (set in `Pi05.extra_export_args`)

This matches our findings exactly.

## Recommendations

### Short-term (current OV 2026.1)

1. **Keep the dual-path strategy** — direct by default, `via_onnx=True` per-policy override. No architectural changes needed.
2. **Add a conversion validation step** that compares OV output to PyTorch reference and warns if max abs diff exceeds a threshold (e.g., 1.0). Pi0.5's 5.15 divergence should trigger a warning.
3. **Do not default to torch.export** — it fails for both VLM policies. It only works for simple models where direct also works.

### Medium-term (file OV issues)

4. **File OV bug: `aten.empty_permuted.default`** — blocks torch.export for all HuggingFace transformer models.
5. **File OV bug: `aten::cat` with empty/rank-0 tensors** — blocks direct conversion for models with KV cache (PaLIGemma, Gemma).
6. **File OV bug: `aten.normal.float_float`** — blocks torch.export for any model using stochastic sampling (flow matching, diffusion).

### Long-term

7. **OV-CPU is non-viable for VLM deployment.** The 14-309x gap vs PyTorch-CUDA means that without an OV GPU plugin (Intel Arc/dGPU) or fundamental CPU plugin optimisation for large attention graphs, deploying SmolVLA/Pi0.5 via OpenVINO on CPU is impractical. Target Intel GPU/NPU devices or alternative runtimes for these models.
8. **Re-benchmark when OV fixes the above ops** — torch.export may then become viable and could produce better-optimised IR than TorchScript tracing.

## Reproducing

```bash
# Full benchmark (CPU + CUDA PyTorch baseline, OV-CPU inference)
uv run --no-project python benchmark_ov_conversion.py

# Single policy
uv run --no-project python benchmark_ov_conversion.py --policies ACT

# CPU only (no CUDA baseline)
uv run --no-project python benchmark_ov_conversion.py --pt-devices cpu --output cpu_only.json
```

## Appendix: Failure Error Messages

### SmolVLA — torch.export
```
OpConversionFailure: Model wasn't fully converted.
-- No conversion rule found for operations: aten.empty_permuted.default
```

### SmolVLA — via ONNX
```
[ONNX Frontend] Conversion failed for Where-20
While validating ONNX node '<Node(Where): index_put>':
Argument 1 and 2 element types must match.
(opset1::Select (boolean[1,1024], i64[1024], f32[1,1024]))
```

### Pi0.5 — Direct (TorchScript)
```
SequenceMark / aten::cat
While validating node 'opset1::Concat Concat_1902944':
Axis -2 out of the tensor rank range [-1, 0].
Inputs: bf16[0] (empty KV cache) + bf16[?,?,256..,256]
```

### Pi0.5 — torch.export (FX)
```
SequenceMark / aten.cat.default — same axis failure
-- No conversion rule found for operations: aten.normal.float_float
```
