#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark OpenVINO conversion methods for Physical AI Studio policies.

Compares three conversion paths:
1. Direct: ov.convert_model(torch.nn.Module)        — TorchScript tracing
2. torch.export: torch.export.export() → ov.convert_model(ExportedProgram) — FX graph
3. Via ONNX: torch.onnx.export() → ov.convert_model("model.onnx")         — two-step

Measures: conversion time, peak memory, output IR size, inference latency,
          and numerical accuracy vs PyTorch reference.

PyTorch baseline runs on both CPU and CUDA (if available).
OpenVINO inference runs on CPU only (no NVIDIA GPU plugin in OV).
Conversion is always performed from a CPU copy of the model (OV/ONNX paths
require CPU tensors); PyTorch latency on CUDA is reported separately.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    """Return peak RSS in MB (Linux only)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB


def _reset_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_inputs(input_dict: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in input_dict.items()}


@dataclass
class BenchmarkResult:
    policy_name: str
    method: str
    success: bool
    error: str | None = None
    conversion_time_s: float | None = None
    peak_rss_mb: float | None = None
    ir_size_mb: float | None = None
    inference_latency_ms: float | None = None
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None


# ---------------------------------------------------------------------------
# Model factories — build each policy with dummy dataset_stats
# ---------------------------------------------------------------------------

def _make_act_policy():
    """Create an ACT policy with dummy dataset stats."""
    from physicalai.data import FeatureType

    dataset_stats = {
        "observation.state": {
            "name": "observation.state",
            "type": FeatureType.STATE,
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        },
        "observation.images.top": {
            "name": "observation.images.top",
            "type": FeatureType.VISUAL,
            "shape": (3, 480, 640),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "action": {
            "name": "action",
            "type": FeatureType.ACTION,
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        },
    }

    from physicalai.policies import ACT
    policy = ACT(dataset_stats=dataset_stats, chunk_size=10, n_action_steps=10)
    policy.eval()
    return policy


def _make_smolvla_policy():
    """Create a SmolVLA policy with dummy dataset stats."""
    from physicalai.data import FeatureType

    dataset_stats = {
        "observation.state": {
            "name": "observation.state",
            "type": str(FeatureType.STATE),
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        },
        "observation.images.top": {
            "name": "observation.images.top",
            "type": str(FeatureType.VISUAL),
            "shape": (3, 480, 640),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "action": {
            "name": "action",
            "type": str(FeatureType.ACTION),
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
        },
    }

    from physicalai.policies import SmolVLA
    policy = SmolVLA(
        dataset_stats=dataset_stats,
        chunk_size=10,
        n_action_steps=10,
        load_vlm_weights=False,
    )
    policy.eval()
    return policy


def _make_pi05_policy():
    """Create a Pi0.5 policy with dummy dataset stats."""
    from physicalai.data import FeatureType

    dataset_stats = {
        "observation.state": {
            "name": "observation.state",
            "type": str(FeatureType.STATE),
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        },
        "observation.images.top": {
            "name": "observation.images.top",
            "type": str(FeatureType.VISUAL),
            "shape": (3, 224, 224),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "action": {
            "name": "action",
            "type": str(FeatureType.ACTION),
            "shape": (14,),
            "mean": [0.0] * 14,
            "std": [1.0] * 14,
            "q01": [-1.0] * 14,
            "q99": [1.0] * 14,
        },
    }

    from physicalai.policies import Pi05
    policy = Pi05(
        dataset_stats=dataset_stats,
        chunk_size=10,
        n_action_steps=10,
        gradient_checkpointing=False,
        compile_model=False,
    )
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Conversion methods (always run on CPU model+inputs)
# ---------------------------------------------------------------------------

def _get_model_and_input(policy):
    """Extract the inner model and its preprocessed sample input (CPU)."""
    sample = policy.model.sample_input
    processed = policy._preprocessor(sample)
    input_dict = {k: v.cpu() for k, v in processed.items() if isinstance(v, torch.Tensor)}
    return policy.model, input_dict


def _get_forward_arg_name(model) -> str:
    """Get the first positional arg name of model.forward (excluding self)."""
    import inspect
    sig = inspect.signature(model.forward)
    for name, param in sig.parameters.items():
        if name != "self" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ):
            return name
    return "batch"


def _pytorch_inference(model, input_dict, arg_name) -> np.ndarray:
    """Run a single PyTorch forward, return numpy output (always cast to fp32 cpu)."""
    with torch.no_grad():
        out = model(**{arg_name: input_dict})
    if isinstance(out, tuple):
        out = out[0]
    return out.detach().float().cpu().numpy()


def _measure_pytorch_latency(model, input_dict, arg_name, device: torch.device, n_runs: int = 20) -> float:
    """Return per-step latency in ms. Uses CUDA events when on GPU."""
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(3):  # warmup
            model(**{arg_name: input_dict})
        if use_cuda:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(n_runs):
                model(**{arg_name: input_dict})
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / n_runs  # ms
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(**{arg_name: input_dict})
        return (time.perf_counter() - t0) / n_runs * 1000


def convert_direct(model, input_dict, arg_name, tmp_dir):
    """Method 1: ov.convert_model(torch.nn.Module)."""
    input_shapes = [openvino.Shape(tuple(t.shape)) for t in input_dict.values()]
    ov_model = openvino.convert_model(
        model,
        example_input={arg_name: input_dict},
        input=input_shapes,
    )
    xml_path = os.path.join(tmp_dir, "direct.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=False)
    return ov_model, xml_path


def convert_torch_export(model, input_dict, arg_name, tmp_dir):
    """Method 2: torch.export.export() → ov.convert_model(ExportedProgram)."""
    exported = torch.export.export(model, args=(), kwargs={arg_name: input_dict})
    ov_model = openvino.convert_model(exported)
    xml_path = os.path.join(tmp_dir, "torch_export.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=False)
    return ov_model, xml_path


def convert_via_onnx(model, input_dict, arg_name, tmp_dir):
    """Method 3: torch.onnx.export → ov.convert_model(onnx)."""
    onnx_path = os.path.join(tmp_dir, "model.onnx")
    torch.onnx.export(
        model,
        args=(),
        kwargs={arg_name: input_dict},
        f=onnx_path,
        input_names=list(input_dict.keys()),
        output_names=["action"],
    )
    input_shapes = [openvino.Shape(tuple(t.shape)) for t in input_dict.values()]
    ov_model = openvino.convert_model(
        onnx_path,
        input=input_shapes,
    )
    xml_path = os.path.join(tmp_dir, "via_onnx.xml")
    openvino.save_model(ov_model, xml_path, compress_to_fp16=False)
    return ov_model, xml_path


# ---------------------------------------------------------------------------
# OV Inference
# ---------------------------------------------------------------------------

def _run_ov_inference(ov_model, input_dict_cpu, ov_device: str = "CPU") -> tuple[np.ndarray, float]:
    """Compile and run OV inference, return output numpy and latency (ms)."""
    core = openvino.Core()
    compiled = core.compile_model(ov_model, ov_device)
    infer_req = compiled.create_infer_request()

    feed = {i: tensor.cpu().numpy() for i, (_, tensor) in enumerate(input_dict_cpu.items())}

    # Warm-up
    for _ in range(3):
        infer_req.infer(feed)

    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        infer_req.infer(feed)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    result = infer_req.get_output_tensor(0).data.copy()
    return result, elapsed


def _ir_size_mb(xml_path: str) -> float:
    """Total size of .xml + .bin in MB."""
    xml_p = Path(xml_path)
    bin_p = xml_p.with_suffix(".bin")
    total = xml_p.stat().st_size
    if bin_p.exists():
        total += bin_p.stat().st_size
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

METHODS = {
    "direct (TorchScript)": convert_direct,
    "torch.export (FX)": convert_torch_export,
    "via ONNX": convert_via_onnx,
}


def benchmark_policy(
    policy_name: str,
    policy_factory,
    pt_devices: list[str],
    skip_methods: set[str] | None = None,
) -> tuple[list[BenchmarkResult], dict[str, float]]:
    """Benchmark all conversion methods for a single policy.

    Returns (results, pt_latency_per_device).
    """
    skip_methods = skip_methods or set()
    results: list[BenchmarkResult] = []
    pt_latency: dict[str, float] = {}

    print(f"\n{'='*70}")
    print(f"  Policy: {policy_name}")
    print(f"{'='*70}")

    try:
        policy = policy_factory()
    except Exception as e:
        print(f"  FAILED to create policy: {e}")
        for method_name in METHODS:
            results.append(BenchmarkResult(
                policy_name=policy_name, method=method_name,
                success=False, error=f"Policy creation failed: {e}",
            ))
        return results, pt_latency

    model, input_dict_cpu = _get_model_and_input(policy)
    arg_name = _get_forward_arg_name(model)

    # PyTorch reference (CPU, fp32 numpy)
    print("  Running PyTorch reference inference (CPU)...")
    model.eval()
    ref_output = _pytorch_inference(model, input_dict_cpu, arg_name)
    print(f"  Reference output shape: {ref_output.shape}")

    # PyTorch latency on each requested device
    for dev_name in pt_devices:
        device = torch.device(dev_name)
        try:
            if dev_name == "cuda":
                model_dev = model.to(device)
                inp_dev = _move_inputs(input_dict_cpu, device)
            else:
                model_dev = model  # already CPU
                inp_dev = input_dict_cpu
            lat = _measure_pytorch_latency(model_dev, inp_dev, arg_name, device)
            pt_latency[dev_name] = lat
            print(f"  PyTorch {dev_name.upper()} latency: {lat:.1f} ms")
        except Exception as e:
            print(f"  PyTorch {dev_name.upper()} latency FAILED: {e}")
            pt_latency[dev_name] = float("nan")
        finally:
            if dev_name == "cuda":
                # Move model back to CPU for conversion
                model = model.to("cpu")
                _reset_memory()

    # Conversions (always from CPU model + CPU inputs)
    for method_name, convert_fn in METHODS.items():
        print(f"\n  --- {method_name} ---")
        _reset_memory()
        result = BenchmarkResult(policy_name=policy_name, method=method_name, success=False)

        if method_name in skip_methods:
            result.error = "skipped via --skip"
            print(f"    SKIPPED: {result.error}")
            results.append(result)
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                rss_before = _peak_rss_mb()
                t0 = time.perf_counter()

                ov_model, xml_path = convert_fn(model, input_dict_cpu, arg_name, tmp_dir)

                result.conversion_time_s = time.perf_counter() - t0
                result.peak_rss_mb = _peak_rss_mb() - rss_before
                result.ir_size_mb = _ir_size_mb(xml_path)

                print(f"    Conversion: {result.conversion_time_s:.2f}s | "
                      f"IR size: {result.ir_size_mb:.2f} MB | "
                      f"Peak RSS delta: {result.peak_rss_mb:.1f} MB")

                ov_output, latency = _run_ov_inference(ov_model, input_dict_cpu, "CPU")
                result.inference_latency_ms = latency

                ov_flat = ov_output.flatten()
                ref_flat = ref_output.flatten()
                min_len = min(len(ov_flat), len(ref_flat))
                if min_len > 0:
                    diff = np.abs(ov_flat[:min_len] - ref_flat[:min_len])
                    result.max_abs_diff = float(np.max(diff))
                    result.mean_abs_diff = float(np.mean(diff))
                else:
                    result.max_abs_diff = float("nan")
                    result.mean_abs_diff = float("nan")

                result.success = True
                print(f"    Inference: {result.inference_latency_ms:.2f} ms | "
                      f"Max diff: {result.max_abs_diff:.6f} | "
                      f"Mean diff: {result.mean_abs_diff:.6f}")

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            print(f"    FAILED: {result.error}")
            traceback.print_exc(limit=3)

        results.append(result)

    del policy, model
    _reset_memory()
    return results, pt_latency


def print_report(
    all_results: list[BenchmarkResult],
    pt_latency_map: dict[str, dict[str, float]],
) -> None:
    """Print formatted benchmark report."""
    print(f"\n\n{'='*100}")
    print("  OPENVINO CONVERSION BENCHMARK REPORT")
    print(f"{'='*100}")
    print(f"  OpenVINO: {openvino.__version__}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'})")
    print(f"{'='*100}\n")

    policies: dict[str, list[BenchmarkResult]] = {}
    for r in all_results:
        policies.setdefault(r.policy_name, []).append(r)

    for policy_name, results in policies.items():
        print(f"  Policy: {policy_name}")
        for dev, lat in pt_latency_map.get(policy_name, {}).items():
            print(f"    PyTorch {dev.upper()} latency: {lat:.1f} ms")
        print()

        header = (f"  {'Method':<25} {'Status':<8} {'Conv(s)':<10} {'IR(MB)':<10} "
                  f"{'OV-CPU(ms)':<12} {'MaxDiff':<12} {'MeanDiff':<12} {'RSS(MB)':<10}")
        print(header)
        print(f"  {'-'*111}")

        for r in results:
            status = "OK" if r.success else "FAIL"
            conv = f"{r.conversion_time_s:.2f}" if r.conversion_time_s is not None else "N/A"
            ir = f"{r.ir_size_mb:.2f}" if r.ir_size_mb is not None else "N/A"
            lat = f"{r.inference_latency_ms:.2f}" if r.inference_latency_ms is not None else "N/A"
            maxd = f"{r.max_abs_diff:.6f}" if r.max_abs_diff is not None else "N/A"
            meand = f"{r.mean_abs_diff:.6f}" if r.mean_abs_diff is not None else "N/A"
            rss = f"{r.peak_rss_mb:.1f}" if r.peak_rss_mb is not None else "N/A"
            print(f"  {r.method:<25} {status:<8} {conv:<10} {ir:<10} {lat:<12} {maxd:<12} {meand:<12} {rss:<10}")

            if r.error:
                print(f"    Error: {r.error}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", nargs="+",
                        choices=["ACT", "SmolVLA", "Pi0.5"],
                        default=["ACT", "SmolVLA", "Pi0.5"],
                        help="Policies to benchmark")
    parser.add_argument("--pt-devices", nargs="+",
                        choices=["cpu", "cuda"],
                        default=None,
                        help="PyTorch devices to measure latency on. "
                             "Default: cpu (+ cuda if available).")
    parser.add_argument("--skip-methods", nargs="*", default=[],
                        choices=list(METHODS),
                        help="Conversion methods to skip")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    if args.pt_devices is None:
        args.pt_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    factories = {
        "ACT": _make_act_policy,
        "SmolVLA": _make_smolvla_policy,
        "Pi0.5": _make_pi05_policy,
    }

    all_results: list[BenchmarkResult] = []
    pt_latency_map: dict[str, dict[str, float]] = {}

    for policy_name in args.policies:
        try:
            results, pt_lat = benchmark_policy(
                policy_name, factories[policy_name],
                pt_devices=args.pt_devices,
                skip_methods=set(args.skip_methods),
            )
        except Exception as e:
            print(f"FATAL: benchmark_policy crashed for {policy_name}: {e}")
            traceback.print_exc()
            results = [BenchmarkResult(
                policy_name=policy_name, method=m,
                success=False, error=f"Policy crashed: {e}",
            ) for m in METHODS]
            pt_lat = {}
        all_results.extend(results)
        pt_latency_map[policy_name] = pt_lat

    print_report(all_results, pt_latency_map)

    json_results = [{
        "policy": r.policy_name,
        "method": r.method,
        "success": r.success,
        "error": r.error,
        "conversion_time_s": r.conversion_time_s,
        "peak_rss_delta_mb": r.peak_rss_mb,
        "ir_size_mb": r.ir_size_mb,
        "ov_cpu_inference_latency_ms": r.inference_latency_ms,
        "max_abs_diff": r.max_abs_diff,
        "mean_abs_diff": r.mean_abs_diff,
    } for r in all_results]

    with open(args.output, "w") as f:
        json.dump({
            "openvino_version": openvino.__version__,
            "pytorch_version": torch.__version__,
            "python_version": ".".join(map(str, sys.version_info[:3])),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ov_inference_device": "CPU",
            "pt_latency_ms": pt_latency_map,
            "results": json_results,
        }, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
