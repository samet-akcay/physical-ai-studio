#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation script for the InferenceRunner refactor.

Validates the complete train → export → inference pipeline with
numerical consistency checks across all export backends.

This script verifies that the runner refactor (strategies → runners)
does not break any existing functionality:
1. Train ACT policy with fast_dev_run=1
2. Export to all supported backends (openvino, onnx, torch_export_ir)
3. Load exported model with InferenceModel
4. Compare inference output vs PyTorch policy output numerically

Usage:
    python tests/integration/validate_runner_refactor.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from physicalai.data import LeRobotDataModule
from physicalai.data.lerobot import FormatConverter
from physicalai.inference import InferenceModel
from physicalai.policies import get_policy
from physicalai.train import Trainer

EXPORT_BACKENDS = ["openvino", "onnx", "torch_export_ir"]


def train_policy():
    """Train ACT policy with fast_dev_run on lerobot/pusht dataset."""
    print("=" * 60)
    print("STEP 1: Training ACT policy (fast_dev_run=1)")
    print("=" * 60)

    datamodule = LeRobotDataModule(
        repo_id="lerobot/pusht",
        train_batch_size=8,
        episodes=list(range(10)),
        video_backend="pyav",
    )
    policy = get_policy("act", source="physicalai")
    trainer = Trainer(
        fast_dev_run=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    trainer.fit(policy, datamodule=datamodule)
    assert trainer.state.finished, "Training did not finish successfully"

    print("  ✓ Training completed successfully")
    return policy, datamodule


def get_reference_output(policy, datamodule):
    """Get reference output from trained PyTorch policy."""
    print("\n" + "=" * 60)
    print("STEP 2: Getting reference output from PyTorch policy")
    print("=" * 60)

    sample_batch = next(iter(datamodule.train_dataloader()))
    batch_observation = FormatConverter.to_observation(sample_batch)
    single_observation = batch_observation[0:1].to("cpu")

    torch.manual_seed(42)
    policy.eval()
    with torch.no_grad():
        train_action = policy.predict_action_chunk(single_observation)
    if isinstance(train_action, tuple):
        train_action = train_action[0]
    train_action = train_action.squeeze(0)
    if len(train_action.shape) > 1:
        train_action = train_action[0]

    print(f"  ✓ Reference action shape: {train_action.shape}")
    print(f"  ✓ Reference action values: {train_action[:5]}...")
    return single_observation, train_action


def validate_export_and_inference(policy, single_observation, reference_action, export_dir):
    """Export to all backends and validate numerical consistency."""
    print("\n" + "=" * 60)
    print("STEP 3: Export → Load → Inference → Numerical Comparison")
    print("=" * 60)

    results = {}

    for backend in EXPORT_BACKENDS:
        print(f"\n  --- Backend: {backend} ---")
        backend_dir = export_dir / f"act_{backend}"

        policy.export(backend_dir, backend)
        assert backend_dir.exists(), f"Export directory not created for {backend}"
        assert (backend_dir / "metadata.yaml").exists(), f"metadata.yaml missing for {backend}"
        print(f"  ✓ Exported to {backend_dir}")

        inference_model = InferenceModel.load(backend_dir)
        assert inference_model.backend.value == backend, (
            f"Backend mismatch: expected {backend}, got {inference_model.backend.value}"
        )
        print(f"  ✓ Loaded InferenceModel (backend={inference_model.backend.value})")
        print(f"  ✓ Runner: {inference_model.runner!r}")

        torch.manual_seed(42)
        inference_input = single_observation.to_numpy().to_dict(flatten=False)
        inference_output = inference_model.select_action(inference_input)
        inference_tensor = torch.as_tensor(inference_output).cpu().squeeze(0)
        if len(inference_tensor.shape) > 1:
            inference_tensor = inference_tensor[0]

        print(f"  ✓ Inference action shape: {inference_tensor.shape}")
        print(f"  ✓ Inference action values: {inference_tensor[:5]}...")

        try:
            torch.testing.assert_close(inference_tensor, reference_action, rtol=0.2, atol=0.2)
            print(f"  ✓ Numerical consistency PASSED (rtol=0.2, atol=0.2)")
            results[backend] = "PASS"
        except AssertionError as e:
            print(f"  ✗ Numerical consistency FAILED: {e}")
            results[backend] = f"FAIL: {e}"

    return results


def validate_runner_repr(policy, export_dir):
    """Validate that runner __repr__ works correctly."""
    print("\n" + "=" * 60)
    print("STEP 4: Validating runner repr and reset")
    print("=" * 60)

    backend_dir = export_dir / "act_openvino"
    if not backend_dir.exists():
        policy.export(backend_dir, "openvino")

    model = InferenceModel.load(backend_dir)

    repr_str = repr(model)
    assert "InferenceModel" in repr_str, f"Expected 'InferenceModel' in repr, got: {repr_str}"
    print(f"  ✓ InferenceModel repr: {repr_str}")

    runner_repr = repr(model.runner)
    print(f"  ✓ Runner repr: {runner_repr}")

    model.reset()
    print("  ✓ Reset completed without errors")


def main():
    """Run the full validation pipeline."""
    print("\n" + "=" * 60)
    print("  INFERENCE RUNNER REFACTOR - VALIDATION SCRIPT")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_dir = Path(tmp_dir)

        policy, datamodule = train_policy()
        single_observation, reference_action = get_reference_output(policy, datamodule)
        results = validate_export_and_inference(policy, single_observation, reference_action, export_dir)
        validate_runner_repr(policy, export_dir)

        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        all_passed = True
        for backend, result in results.items():
            status = "✓" if result == "PASS" else "✗"
            print(f"  {status} {backend}: {result}")
            if result != "PASS":
                all_passed = False

        if all_passed:
            print("\n  ✓ ALL VALIDATIONS PASSED - Runner refactor is safe")
            return 0

        print("\n  ✗ SOME VALIDATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
