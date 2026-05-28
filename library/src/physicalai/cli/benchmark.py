# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: INP001

"""Register the ``physicalai benchmark`` subcommand."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Protocol

from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from physicalai.cli._spec import SubcommandSpec  # noqa: PLC2701


class _WritableResults(Protocol):
    """Protocol for benchmark result persistence helpers."""

    def summary(self) -> str:
        """Return a human-readable summary string."""

    def to_json(self, path: Path) -> None:
        """Write results as JSON."""

    def to_csv(self, path: Path) -> None:
        """Write results as CSV."""


logger = logging.getLogger(__name__)

HELP = "Run benchmark evaluation."


def build_parser() -> ArgumentParser:
    """Build the ``benchmark`` parser.

    Returns:
        Parser for ``physicalai benchmark``.
    """
    from physicalai.benchmark.gyms import Benchmark  # noqa: PLC0415

    parser = ArgumentParser(prog="physicalai benchmark", description="Run benchmark evaluation on a trained policy.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML/JSON config file.")
    parser.add_subclass_arguments(Benchmark, "benchmark", required=True)
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Policy class path (e.g., physicalai.policies.ACT).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to checkpoint file (.ckpt) or export directory. "
            "For Policy subclasses: path to Lightning checkpoint (.ckpt). "
            "For InferenceModel: path to export directory containing model files. "
            "If not provided, uses randomly initialized policy (Policy only)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/benchmark",
        help="Directory to save benchmark results.",
    )
    return parser


def _load_policy(policy_path: str, ckpt_path: str | None) -> tuple[Any, str]:
    """Load policy from class path and optional checkpoint.

    Args:
        policy_path: Fully qualified policy class path.
        ckpt_path: Path to checkpoint file or export directory.

    Returns:
        Tuple of instantiated policy and selected device string.

    Raises:
        ImportError: If the policy class cannot be imported.
        ValueError: If ``InferenceModel`` is used without ``ckpt_path``.
    """
    from physicalai.devices import get_available_device  # noqa: PLC0415
    from physicalai.inference import InferenceModel  # noqa: PLC0415

    module_path, class_name = policy_path.rsplit(".", 1)
    try:
        policy_class = getattr(importlib.import_module(module_path), class_name)
    except (ImportError, AttributeError) as exc:
        msg = f"Could not import policy class '{policy_path}'"
        raise ImportError(msg) from exc

    is_inference_model = policy_class is InferenceModel or (
        isinstance(policy_class, type) and issubclass(policy_class, InferenceModel)
    )

    if is_inference_model:
        if not ckpt_path:
            msg = "InferenceModel requires --ckpt_path pointing to export directory"
            raise ValueError(msg)
        policy = InferenceModel.load(ckpt_path)
        return policy, policy.device

    device = get_available_device()
    if ckpt_path:
        policy = policy_class.load_from_checkpoint(ckpt_path)
    else:
        logger.warning("No checkpoint provided - using randomly initialized policy")
        policy = policy_class()

    policy.to(device)
    policy.eval()
    return policy, device


def run(parser: ArgumentParser, cfg: Namespace) -> int:
    """Dispatch ``benchmark`` by instantiating benchmark and evaluating policy.

    Args:
        parser: Parser used to instantiate the benchmark.
        cfg: Parsed configuration namespace.

    Returns:
        Process exit code.
    """
    benchmark = parser.instantiate_classes(Namespace(benchmark=cfg.benchmark)).benchmark
    policy, device = _load_policy(cfg.policy, cfg.ckpt_path)

    logger.info("Benchmark: %s", benchmark)
    logger.info("Policy: %s", type(policy).__name__)
    logger.info("Device: %s", device)

    try:
        results = benchmark.evaluate(policy=policy)
        _write_results(results, cfg.output_dir)
    finally:
        for gym in benchmark.gyms:
            gym.close()

    return 0


def _write_results(results: _WritableResults, output_dir: str) -> None:
    """Print and persist benchmark results.

    Args:
        results: Benchmark results object with summary and export helpers.
        output_dir: Directory where results files should be written.
    """
    print(results.summary())  # noqa: T201

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_json(output_path / "results.json")
    results.to_csv(output_path / "results.csv")
    logger.info("Results saved to %s", output_path)


def register() -> SubcommandSpec:
    """Return the ``benchmark`` subcommand spec.

    Returns:
        Registered spec for the shared CLI host.
    """
    return SubcommandSpec(name="benchmark", parser=build_parser(), dispatch=run, help=HELP)
