# OpenVINO Conversion Benchmark — Findings Report

## Environment

| Component | Old run | New run |
|-----------|---------|---------|
| OpenVINO | 2026.1.0-21367 | 2026.1.0-21367 (unchanged) |
| PyTorch | 2.10.0+cu128 | **2.12.0+cu130** |
| Python | 3.12.8 | 3.12.3 |
| OV inference device | CPU | CPU (no Intel GPU on host) |
| PyTorch device(s) | CPU | CPU + **CUDA (RTX A6000)** |

OpenVINO has no plugin for NVIDIA GPUs, so OV inference stays on CPU. PyTorch runs on both CPU and CUDA so the OV-CPU IR can be compared against a realistic GPU baseline.

Raw data: [benchmark_results_torch210.json](file:///home/sakcay/projects/physical-ai/physical-ai-studio-worktrees/ov-conversion-benchmark/benchmark_results_torch210.json), [benchmark_results_torch212.json](file:///home/sakcay/projects/physical-ai/physical-ai-studio-worktrees/ov-conversion-benchmark/benchmark_results_torch212.json) (also copied to `benchmark_results.json`).

## TL;DR

**PyTorch 2.12 fixed nothing on the OpenVINO conversion side.** Every failure that blocked SmolVLA/Pi0.5 in 2.10 reproduces verbatim in 2.12 with the same error messages. The compatibility matrix is unchanged.

**OV-CPU latencies got measurably worse** under 2.12 across all working cells (1.4× slower for ACT, 1.78× slower for SmolVLA-direct, 1.92× slower for Pi0.5-via-ONNX). Conversion times also drift up.

**The CUDA baseline reframes the deployment story.** OV-CPU is now compared against a real GPU number, and the gap is brutal for VLMs: Pi0.5 OV-CPU is 309× slower than Pi0.5 PyTorch-CUDA. ACT is the only model where OV-CPU is competitive with anything (~3× slower than CUDA, ~2.3× faster than CPU).

## Methods Tested

1. **Direct** — `ov.convert_model(nn.Module, example_input=...)` — TorchScript-based tracing
2. **torch.export** — `torch.export.export()` → `ov.convert_model(ExportedProgram)` — FX graph capture
3. **via ONNX** — `torch.onnx.export()` → `ov.convert_model("model.onnx")` — two-step through ONNX

## Compatibility Matrix (PT 2.10 vs PT 2.12)

| Policy | Direct | torch.export | via ONNX |
|--------|--------|--------------|----------|
| **ACT** (~33M) | OK to OK | OK to OK | OK to OK |
| **SmolVLA** (~500M) | OK to OK | FAIL to FAIL | FAIL to FAIL |
| **Pi0.5** (~3B) | FAIL to FAIL | FAIL to FAIL | OK to OK |

Identical. **No single conversion method works for all three policies under either PyTorch version.**

## Detailed Results — PyTorch 2.12 + CUDA baseline

### ACT (Action Chunking with Transformers, ~33M params)

PyTorch latency: **CPU 66.6 ms** · **CUDA 8.5 ms**

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | 5.74 | 131.43 | 28.78 | **2.31x faster** | 3.39x slower | 0.72 |
| torch.export | 8.65 | 126.20 | 33.48 | **1.99x faster** | 3.94x slower | 0.72 |
| via ONNX | 14.48 | 126.54 | 28.95 | **2.30x faster** | 3.41x slower | 0.72 |

ACT remains the only model where OV-CPU beats PyTorch-CPU. CUDA is still ~3.4x faster than OV-CPU.

### SmolVLA (Vision-Language-Action, SmolVLM2-500M backbone, ~500M params)

PyTorch latency: **CPU 898 ms** · **CUDA 235 ms**

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | 76.96 | 956.71 | **3287** | **3.66x slower** | **14.0x slower** | 0.0043 |
| torch.export | FAIL | — | — | — | — | — |
| via ONNX | FAIL | — | — | — | — | — |

Only direct conversion succeeds. OV-CPU regressed badly: from 1845 ms (PT 2.10) to 3287 ms (PT 2.12) — a **1.78x slowdown** for the same model and IR layout. PT-CPU also got slower (756 -> 898 ms) but the OV side moved more.

Failures (unchanged from 2.10):
- **torch.export**: `aten.empty_permuted.default has no OV conversion rule` (HuggingFace SDPA attention)
- **via ONNX**: `Where-20: Argument 1 and 2 element types must match` (i64 vs f32 in `index_put`)

### Pi0.5 (PaLIGemma + flow-matching action expert, ~3B params)

PyTorch latency: **CPU 8464 ms** · **CUDA 252 ms** (33.6x CUDA speedup)

| Method | Conv (s) | IR (MB) | OV-CPU (ms) | vs PT-CPU | vs PT-CUDA | Max Abs Diff |
|--------|---------:|--------:|------------:|----------:|-----------:|-------------:|
| direct | FAIL | — | — | — | — | — |
| torch.export | FAIL | — | — | — | — | — |
| via ONNX | **930** | **6763** | **77919** | **9.2x slower** | **309x slower** | **5.15** |

Only via-ONNX succeeds, and the result has gotten worse on every axis vs 2.10:
- Conversion: 808 s -> **930 s** (+15 %)
- OV-CPU latency: 40.6 s -> **77.9 s** per step (+92 %)
- Max abs diff: 4.25 -> **5.15** (+21 %)

The 309x gap vs PyTorch CUDA confirms OV-CPU is non-viable as a deployment target for Pi0.5.

Failures (unchanged from 2.10):
- **direct**: `aten::cat Axis -2 out of tensor rank range [-1, 0]` (KV cache concat with empty rank-0 tensor) + `SequenceMark` internal type
- **torch.export**: same `aten.cat.default` axis failure + `aten.normal.float_float has no conversion rule`

## PT 2.10 -> PT 2.12 deltas (working cells only)

| Cell | OV-CPU lat (2.10) | OV-CPU lat (2.12) | Delta | Conv (2.10) | Conv (2.12) | Delta |
|---|---:|---:|---:|---:|---:|---:|
| ACT direct | 20.4 ms | 28.8 ms | **+41 %** | 4.4 s | 5.7 s | +30 % |
| ACT torch.export | 21.7 ms | 33.5 ms | **+54 %** | 7.8 s | 8.6 s | +10 % |
| ACT via ONNX | 19.6 ms | 29.0 ms | **+48 %** | 13.7 s | 14.5 s | +6 % |
| SmolVLA direct | 1845 ms | 3287 ms | **+78 %** | 75.3 s | 77.0 s | +2 % |
| Pi0.5 via ONNX | 40625 ms | 77919 ms | **+92 %** | 808.7 s | 930.1 s | +15 % |

Every successful cell got slower at OV inference time. Conversion-time impact is small except for ACT (+10–30 %). IR sizes are essentially identical between versions, so the IR layout did not change — the regression sits inside the OpenVINO runtime / PyTorch graph conversion path under 2.12, not in the produced model topology.

## Key Findings

### 1. PT 2.12 does not unblock any OV conversion failure

Every blocker observed in the PT 2.10 run reproduces with **the same op and same error message** under PT 2.12:

| Architecture pattern | Blocker | Affected | Status in 2.12 |
|---|---|---|---|
| HF SDPA attention | `aten.empty_permuted.default` unsupported | SmolVLA torch.export, SmolVLA via ONNX | unchanged |
| PaLIGemma KV cache (empty tensor concat) | `aten::cat` axis validation on rank-0 | Pi0.5 direct, Pi0.5 torch.export | unchanged |
| Flow-matching noise sampling | `aten.normal.float_float` unsupported | Pi0.5 torch.export | unchanged |
| Mixed-type indexing | ONNX Where-20 type mismatch | SmolVLA via ONNX | unchanged |

These are **OpenVINO frontend gaps**, not PyTorch issues. Upgrading PyTorch was never going to fix them; upgrading OpenVINO is what we need.

### 2. PT 2.12 makes OV-CPU inference worse

The OV-CPU runtime path is slower for every working cell, with VLMs hit hardest (+78 % for SmolVLA, +92 % for Pi0.5). PT-CPU also got slower (e.g. SmolVLA PT-CPU 756 -> 898 ms) so part of the regression is inside torch itself, but the OV-CPU regression is larger than the PT-CPU regression in every case.

Hypothesis: the PT 2.12 graph that OV traces / decomposes hits more dynamic shape paths or unfused subgraphs in the OV CPU plugin. Worth investigating with `OV_CPU_PROFILE=1` before filing anything.

### 3. CUDA baseline shows OV-CPU is non-viable for VLMs

| Policy | PT-CUDA | OV-CPU (best) | Ratio |
|--------|--------:|--------------:|------:|
| ACT | 8.5 ms | 28.78 ms | OV-CPU is **3.4x slower** |
| SmolVLA | 235 ms | 3287 ms | OV-CPU is **14x slower** |
| Pi0.5 | 252 ms | 77919 ms | OV-CPU is **309x slower** |

For Pi0.5 the gap is so large that OV-CPU cannot be considered a deployment target without GPU plugin support or a fundamental graph-level redesign. ACT is the only policy where OV-CPU is even in the conversation versus a GPU.

### 4. Pi0.5 numerical accuracy degraded further under 2.12

Pi0.5 via-ONNX max abs diff went from 4.25 to **5.15**. Combined with the doubled inference time, this is the worst result in the suite by every metric. The bf16 -> fp32 promotion in the ONNX path plus the 10-step flow-matching denoising loop amplifies even small per-op differences. This needs validation against trained weights to determine real-world impact on action quality.

### 5. The dual-path strategy in `to_openvino()` remains the right call

Per-policy method selection still matches the data exactly:
- ACT, SmolVLA -> direct
- Pi0.5 -> `via_onnx=True` (already set in `Pi05.extra_export_args`)

No code changes warranted from these results.

## Recommendations

### Short-term (current OV 2026.1)

1. **Do not upgrade PyTorch to 2.12 for OpenVINO export workflows yet.** It fixes no failures and slows OV-CPU inference by 40–90 %. Stay on 2.10 (or 2.11) until the regression is understood.
2. **Keep the dual-path strategy** (direct default + `via_onnx=True` per-policy override). Architecture is correct.
3. **Add a conversion validation step** that compares OV output to PyTorch reference and warns if max abs diff exceeds ~1.0. Pi0.5's 5.15 divergence should fire the warning.

### Medium-term (file/track upstream)

4. **File OV bugs for the four blockers** above. None of them moved between OV 2026.1 + PT 2.10 and OV 2026.1 + PT 2.12, so they are pure OV frontend issues:
   - `aten.empty_permuted.default` (HF SDPA attention path) — blocks torch.export for any HF transformer model
   - `aten::cat` / `aten.cat.default` axis validation on empty rank-0 tensors — blocks direct + torch.export for any KV-cached model
   - `aten.normal.float_float` — blocks torch.export for any stochastic-sampling model (flow matching, diffusion)
   - ONNX `Where-20` type-merge requirement on `i64` vs `f32` Select — blocks via-ONNX for models with mixed-dtype indexing
5. **File the OV-CPU latency regression with PT 2.12** as a separate report. Reproducer is the SmolVLA direct path (1845 -> 3287 ms with no IR changes).

### Long-term

6. **For VLM CPU deployment, OV is currently not the right answer.** Either invest in OV-CPU optimisation for large attention graphs, or target a different runtime (PyTorch CUDA, vLLM, ExecuTorch with hardware delegate).
7. **Re-run this benchmark when the four ops above land in OV.** torch.export then becomes worth a second look.

## Reproducing

```bash
uv run --no-project python benchmark_ov_conversion.py            # PT-CPU + PT-CUDA + OV-CPU, all policies
uv run --no-project python benchmark_ov_conversion.py --policies ACT
uv run --no-project python benchmark_ov_conversion.py --pt-devices cpu --output cpu_only.json
```

Script: [benchmark_ov_conversion.py](file:///home/sakcay/projects/physical-ai/physical-ai-studio-worktrees/ov-conversion-benchmark/benchmark_ov_conversion.py)

## Appendix: PT 2.12 failure error messages

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
(opset1::Select Select_1877721 (boolean[1,1024], i64[1024], f32[1,1024]))
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
