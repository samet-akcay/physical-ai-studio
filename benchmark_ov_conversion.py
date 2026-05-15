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
"""

from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino
import resource
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_rss_mb() -> float:
    """Return peak RSS in MB (Linux only)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB


def _reset_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


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
    from physicalai.data import Feature, FeatureType

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
# Conversion methods
# ---------------------------------------------------------------------------

def _get_model_and_input(policy):
    """Extract the inner model and its preprocessed sample input."""
    sample = policy.model.sample_input
    processed = policy._preprocessor(sample)
    input_dict = {k: v for k, v in processed.items() if isinstance(v, torch.Tensor)}
    return policy.model, input_dict


def _get_forward_arg_name(model):
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


def _pytorch_reference(model, input_dict, arg_name):
    """Run PyTorch reference inference, return numpy output."""
    model.eval()
    with torch.no_grad():
        out = model(**{arg_name: input_dict})
    if isinstance(out, tuple):
        out = out[0]
    return out.cpu().numpy()


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
# Inference & accuracy
# ---------------------------------------------------------------------------

def _run_ov_inference(ov_model, input_dict):
    """Compile and run OV inference, return output numpy and latency."""
    core = openvino.Core()
    compiled = core.compile_model(ov_model, "CPU")
    infer_req = compiled.create_infer_request()

    # Prepare input tensors
    feed = {}
    for i, (name, tensor) in enumerate(input_dict.items()):
        feed[i] = tensor.cpu().numpy()

    # Warm-up
    for _ in range(3):
        infer_req.infer(feed)

    # Timed runs
    n_runs = 20
    t0 = time.perf_counter()
    for _ in range(n_runs):
        infer_req.infer(feed)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms

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


def benchmark_policy(policy_name: str, policy_factory) -> list[BenchmarkResult]:
    """Benchmark all conversion methods for a single policy."""
    results = []

    print(f"\n{'='*70}")
    print(f"  Policy: {policy_name}")
    print(f"{'='*70}")

    # Build policy
    try:
        policy = policy_factory()
    except Exception as e:
        print(f"  FAILED to create policy: {e}")
        for method_name in METHODS:
            results.append(BenchmarkResult(
                policy_name=policy_name, method=method_name,
                success=False, error=f"Policy creation failed: {e}",
            ))
        return results

    model, input_dict = _get_model_and_input(policy)
    arg_name = _get_forward_arg_name(model)

    # PyTorch reference
    print(f"  Running PyTorch reference inference...")
    model.eval()
    ref_output = _pytorch_reference(model, input_dict, arg_name)
    print(f"  Reference output shape: {ref_output.shape}")

    # PyTorch inference latency
    with torch.no_grad():
        # warmup
        for _ in range(3):
            model(**{arg_name: input_dict})
        n_runs = 20
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(**{arg_name: input_dict})
        pt_latency = (time.perf_counter() - t0) / n_runs * 1000
    print(f"  PyTorch CPU latency: {pt_latency:.1f} ms")

    for method_name, convert_fn in METHODS.items():
        print(f"\n  --- {method_name} ---")
        _reset_memory()

        result = BenchmarkResult(policy_name=policy_name, method=method_name, success=False)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Measure conversion
                rss_before = _peak_rss_mb()
                t0 = time.perf_counter()

                ov_model, xml_path = convert_fn(model, input_dict, arg_name, tmp_dir)

                result.conversion_time_s = time.perf_counter() - t0
                result.peak_rss_mb = _peak_rss_mb() - rss_before
                result.ir_size_mb = _ir_size_mb(xml_path)

                print(f"    Conversion: {result.conversion_time_s:.2f}s | "
                      f"IR size: {result.ir_size_mb:.2f} MB | "
                      f"Peak RSS delta: {result.peak_rss_mb:.1f} MB")

                # Inference
                ov_output, latency = _run_ov_inference(ov_model, input_dict)
                result.inference_latency_ms = latency

                # Accuracy
                # Handle shape mismatch — OV may flatten or add dims
                ov_flat = ov_output.flatten()
                ref_flat = ref_output.flatten()
                min_len = min(len(ov_flat), len(ref_flat))
                if min_len > 0:
                    result.max_abs_diff = float(np.max(np.abs(ov_flat[:min_len] - ref_flat[:min_len])))
                    result.mean_abs_diff = float(np.mean(np.abs(ov_flat[:min_len] - ref_flat[:min_len])))
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
            traceback.print_exc()

        results.append(result)

    # Cleanup
    del policy, model
    _reset_memory()

    return results


def print_report(all_results: list[BenchmarkResult], pt_latency_map: dict[str, float]):
    """Print formatted benchmark report."""
    print(f"\n\n{'='*90}")
    print("  OPENVINO CONVERSION BENCHMARK REPORT")
    print(f"{'='*90}")
    print(f"  OpenVINO: {openvino.__version__}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Device:   CPU")
    print(f"{'='*90}\n")

    # Group by policy
    policies = {}
    for r in all_results:
        policies.setdefault(r.policy_name, []).append(r)

    for policy_name, results in policies.items():
        print(f"  Policy: {policy_name}")
        pt_lat = pt_latency_map.get(policy_name, 0)
        print(f"  PyTorch CPU latency: {pt_lat:.1f} ms\n")

        header = f"  {'Method':<25} {'Status':<8} {'Conv(s)':<10} {'IR(MB)':<10} {'Lat(ms)':<10} {'MaxDiff':<12} {'MeanDiff':<12} {'RSS(MB)':<10}"
        print(header)
        print(f"  {'-'*107}")

        for r in results:
            status = "✓" if r.success else "✗"
            conv = f"{r.conversion_time_s:.2f}" if r.conversion_time_s is not None else "N/A"
            ir = f"{r.ir_size_mb:.2f}" if r.ir_size_mb is not None else "N/A"
            lat = f"{r.inference_latency_ms:.2f}" if r.inference_latency_ms is not None else "N/A"
            maxd = f"{r.max_abs_diff:.6f}" if r.max_abs_diff is not None else "N/A"
            meand = f"{r.mean_abs_diff:.6f}" if r.mean_abs_diff is not None else "N/A"
            rss = f"{r.peak_rss_mb:.1f}" if r.peak_rss_mb is not None else "N/A"
            print(f"  {r.method:<25} {status:<8} {conv:<10} {ir:<10} {lat:<10} {maxd:<12} {meand:<12} {rss:<10}")

            if r.error:
                print(f"    Error: {r.error}")

        print()


def main():
    policies = {
        "ACT": _make_act_policy,
        "SmolVLA": _make_smolvla_policy,
        "Pi0.5": _make_pi05_policy,
    }

    all_results = []
    pt_latency_map = {}

    for policy_name, factory in policies.items():
        # Build once to measure PT latency
        try:
            policy = factory()
            model, input_dict = _get_model_and_input(policy)
            arg_name = _get_forward_arg_name(model)
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    model(**{arg_name: input_dict})
                n_runs = 20
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    model(**{arg_name: input_dict})
                pt_latency_map[policy_name] = (time.perf_counter() - t0) / n_runs * 1000
            del policy, model
            _reset_memory()
        except Exception as e:
            print(f"Warning: Could not measure PT latency for {policy_name}: {e}")
            pt_latency_map[policy_name] = 0

        try:
            results = benchmark_policy(policy_name, factory)
        except Exception as e:
            print(f"FATAL: benchmark_policy crashed for {policy_name}: {e}")
            traceback.print_exc()
            results = [BenchmarkResult(
                policy_name=policy_name, method=m,
                success=False, error=f"Policy crashed: {e}",
            ) for m in METHODS]
        all_results.extend(results)

    print_report(all_results, pt_latency_map)

    # Save JSON results
    json_results = []
    for r in all_results:
        json_results.append({
            "policy": r.policy_name,
            "method": r.method,
            "success": r.success,
            "error": r.error,
            "conversion_time_s": r.conversion_time_s,
            "peak_rss_delta_mb": r.peak_rss_mb,
            "ir_size_mb": r.ir_size_mb,
            "inference_latency_ms": r.inference_latency_ms,
            "max_abs_diff": r.max_abs_diff,
            "mean_abs_diff": r.mean_abs_diff,
        })

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "openvino_version": openvino.__version__,
            "pytorch_version": torch.__version__,
            "pytorch_latency_ms": pt_latency_map,
            "results": json_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
