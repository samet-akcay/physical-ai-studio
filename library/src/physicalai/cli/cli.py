# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

r"""Unified CLI for physicalai extending LightningCLI with benchmark support.

This module provides `CLI` - a custom CLI class that extends LightningCLI
to add benchmark evaluation capabilities while preserving all training features.

Design Pattern:
    | Class       | Extends              | Purpose                              |
    |-------------|----------------------|--------------------------------------|
    | Config      | -                    | Training configuration               |
    | DataModule  | LightningDataModule  | Data loading                         |
    | Policy      | LightningModule      | Model definition                     |
    | Trainer     | lightning.Trainer    | Training orchestration               |
    | CLI         | LightningCLI         | Command-line interface               |

Key features:
- All LightningCLI subcommands: fit, validate, test, predict
- Additional ``benchmark`` subcommand for policy evaluation
- Additional ``config`` subcommand for config conversion
- Auto-detection of LeRobot configs in ``fit --config``
- Uses physicalai.train.Trainer by default
- Full class_path support for model (policy), data, and benchmark

Examples:
    # Train with native YAML config
    physicalai fit --config configs/train.yaml

    # Train with a LeRobot config (auto-detected and converted)
    physicalai fit --config path/to/lerobot_config.json

    # Explicitly convert a LeRobot config
    physicalai config --from lerobot path/to/config.json -o train.yaml

    # Override config values from CLI
    physicalai fit \
        --config configs/train.yaml \
        --trainer.max_epochs 200 \
        --data.train_batch_size 64

    # Run benchmark evaluation
    physicalai benchmark --config configs/benchmark/libero.yaml
"""

from __future__ import annotations

import atexit
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from physicalai.data import DataModule
from physicalai.policies.base import Policy
from physicalai.train import Trainer

if TYPE_CHECKING:
    from physicalai.benchmark import Benchmark

logger = logging.getLogger(__name__)

_TRAINING_SUBCOMMANDS = {"fit", "validate", "test", "predict"}


class CLI(LightningCLI):
    r"""Custom CLI extending LightningCLI with benchmark and config support.

    Subcommands:
        - fit, validate, test, predict (inherited from LightningCLI)
        - benchmark: policy evaluation
        - config: config conversion (``--from lerobot``)

    Training subcommands auto-detect LeRobot configs passed via
    ``--config`` and convert them transparently before parsing.

    Example:
        .. code-block:: bash

            # Native config
            physicalai fit --config configs/train.yaml

            # LeRobot config (auto-converted)
            physicalai fit --config lerobot_train_config.json

            # Explicit conversion
            physicalai config --from lerobot config.json -o train.yaml
    """

    def __init__(
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize CLI with benchmark support."""
        # Store benchmark-related state
        self._benchmark: Benchmark | None = None
        self._benchmark_pretrained: str | None = None
        self._benchmark_output_dir: Path | None = None
        self._benchmark_verbose: bool = False
        self._temp_config_file: tempfile.NamedTemporaryFile | None = None

        super().__init__(*args, **kwargs)

        if self._temp_config_file is not None:
            atexit.register(self._cleanup_temp_config)

    def _cleanup_temp_config(self) -> None:
        """Remove temporary converted config file created during auto-detection."""
        if self._temp_config_file is not None:
            Path(self._temp_config_file.name).unlink(missing_ok=True)
            self._temp_config_file = None

    def instantiate_classes(self) -> None:  # noqa: D102
        if self.subcommand in {"benchmark", "config"}:
            return
        super().instantiate_classes()

    @staticmethod
    def subcommands() -> dict[str, set[str]]:  # noqa: D102
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
            "benchmark": {"model", "dataloaders", "datamodule", "trainer"},
            "config": set(),
        }

    _NON_TRAINER_SUBCOMMANDS: ClassVar[set[str]] = {"benchmark", "config"}

    def parse_arguments(self, parser: LightningArgumentParser, args: Any) -> None:  # noqa: ANN401
        """Pre-process args to auto-convert LeRobot configs before parsing."""
        args = self._maybe_convert_lerobot_config(args)

        # Non-trainer subcommands (benchmark, config) use custom parsers that
        # don't populate the parent namespace. Pre-seed the subcommand
        # namespace with its parser defaults so jsonargparse can clone it
        # during argument parsing (empty Namespace gets dropped by merge).
        arg_list: list[str] | None = sys.argv[1:] if args is None else (args if isinstance(args, list) else None)
        if arg_list and arg_list[0] in self._NON_TRAINER_SUBCOMMANDS:
            from jsonargparse import Namespace  # noqa: PLC0415

            subcmd = arg_list[0]
            subparser = self._subcommand_parsers[subcmd]
            sub_defaults = subparser.get_defaults()
            self.config = parser.parse_args(arg_list, namespace=Namespace(**{subcmd: sub_defaults}))
            return

        super().parse_arguments(parser, args)

    def _maybe_convert_lerobot_config(self, args: Any) -> Any:  # noqa: ANN401
        """Detect and convert LeRobot config files in --config arguments.

        Scans the argument list for a training subcommand with a ``--config``
        flag pointing to a LeRobot-format file.  When found, converts the
        file to a temporary native YAML and rewrites the arg list in-place.

        Returns:
            The (possibly modified) argument list, or ``None`` when
            ``sys.argv`` was rewritten in-place.
        """
        arg_list: list[str] | None = None
        if args is None:
            arg_list = sys.argv[1:]
            use_sysargv = True
        elif isinstance(args, list):
            arg_list = list(args)
            use_sysargv = False
        else:
            return args

        if not arg_list or arg_list[0] not in _TRAINING_SUBCOMMANDS:
            return args

        config_indices = [i for i, a in enumerate(arg_list) if a == "--config" and i + 1 < len(arg_list)]
        if not config_indices:
            return args

        from physicalai.config.lerobot import detect_config_format  # noqa: PLC0415

        for idx in config_indices:
            config_path = arg_list[idx + 1]
            if not Path(config_path).exists():
                continue

            try:
                fmt = detect_config_format(config_path)
            except (ValueError, FileNotFoundError):
                continue

            if fmt != "lerobot":
                continue

            logger.info("Detected LeRobot config at %s — auto-converting for physicalai.", config_path)

            from physicalai.config.lerobot import TrainPipelineConfigAdapter  # noqa: PLC0415

            adapter = TrainPipelineConfigAdapter.from_file(config_path)
            self._temp_config_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                mode="w",
                suffix=".yaml",
                prefix="physicalai_converted_",
                delete=False,
                encoding="utf-8",
            )
            adapter.to_yaml(self._temp_config_file.name)
            self._temp_config_file.close()

            arg_list[idx + 1] = self._temp_config_file.name
            logger.info("Using converted config: %s", self._temp_config_file.name)
            break

        if use_sysargv:
            # LightningCLI reads from sys.argv when args=None.  We must
            # mutate sys.argv in-place so the converted path is visible
            # to the downstream parser.  This is safe because CLI
            # instances are short-lived entry-point objects.
            sys.argv[1:] = arg_list
            return None
        return arg_list

    def _add_subcommands(self, parser: LightningArgumentParser, **kwargs: Any) -> None:  # noqa: ANN401
        from lightning.pytorch.cli import (  # noqa: PLC0415
            _get_short_description,  # noqa: PLC2701
            class_from_function,
        )

        self._subcommand_parsers: dict[str, LightningArgumentParser] = {}
        parser_subcommands = parser.add_subcommands()

        trainer_class = (
            self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        )

        for subcommand in self.subcommands():
            if subcommand == "benchmark":
                description = "Run benchmark evaluation on a trained policy."
                subparser_kwargs = kwargs.get(subcommand, {})
                subparser_kwargs.setdefault("description", description)
                subcommand_parser = self._create_benchmark_parser(**subparser_kwargs)
            elif subcommand == "config":
                description = "Convert external config formats (e.g. LeRobot) to physicalai YAML."
                subparser_kwargs = kwargs.get(subcommand, {})
                subparser_kwargs.setdefault("description", description)
                subcommand_parser = self._create_config_parser(**subparser_kwargs)
            else:
                fn = getattr(trainer_class, subcommand)
                description = _get_short_description(fn) or f"Run {subcommand}"
                subparser_kwargs = kwargs.get(subcommand, {})
                subparser_kwargs.setdefault("description", description)
                subcommand_parser = self._prepare_subcommand_parser(trainer_class, subcommand, **subparser_kwargs)

            self._subcommand_parsers[subcommand] = subcommand_parser
            parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    def _create_benchmark_parser(self, **kwargs: Any) -> LightningArgumentParser:  # noqa: ANN401, PLR6301
        from physicalai.benchmark import Benchmark  # noqa: PLC0415

        parser = LightningArgumentParser(**kwargs)

        parser.add_subclass_arguments(
            Benchmark,
            nested_key="benchmark",
            required=True,
            help="Benchmark configuration. Use class_path for specialized benchmarks.",
        )

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
                "If not provided, uses randomly initialized policy."
            ),
        )

        parser.add_argument(
            "--output_dir",
            type=str,
            default="./results/benchmark",
            help="Directory to save benchmark results.",
        )

        return parser

    def _create_config_parser(self, **kwargs: Any) -> LightningArgumentParser:  # noqa: ANN401, PLR6301
        parser = LightningArgumentParser(**kwargs)

        parser.add_argument(
            "--from",
            type=str,
            dest="source_format",
            default="lerobot",
            help="Source config format. Currently supported: 'lerobot'.",
        )

        parser.add_argument(
            "input",
            type=str,
            help="Path to source config file, local checkpoint directory, or HuggingFace Hub repo ID.",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            default=None,
            help="Output YAML file path. Defaults to <input_stem>_physicalai.yaml.",
        )

        return parser

    def _run_subcommand(self, subcommand: str) -> None:
        if subcommand == "benchmark":
            self._run_benchmark()
        elif subcommand == "config":
            self._run_config()
        else:
            super()._run_subcommand(subcommand)

    def _run_config(self) -> None:
        from physicalai.config.lerobot import TrainPipelineConfigAdapter  # noqa: PLC0415

        if self.subcommand is None:
            msg = "No subcommand specified"
            raise ValueError(msg)
        config = self.config.get(self.subcommand)
        if config is None:
            msg = "config subcommand configuration not found"
            raise ValueError(msg)

        source_format = config.source_format
        input_path = config.input
        output_path = config.output

        if source_format != "lerobot":
            msg = f"Unsupported source format: {source_format!r}. Currently supported: 'lerobot'."
            raise ValueError(msg)

        adapter = TrainPipelineConfigAdapter.from_file(input_path)

        if output_path is None:
            stem = Path(input_path).stem if Path(input_path).exists() else "lerobot"
            output_path = f"{stem}_physicalai.yaml"

        result_path = adapter.to_yaml(output_path)
        logger.info("Config written to %s", result_path)
        print(f"Converted config written to: {result_path}")  # noqa: T201

    @staticmethod
    def _load_policy(
        policy_path: str,
        ckpt_path: str | None,
    ) -> tuple[Any, str]:
        """Load policy from class path and optional checkpoint.

        Args:
            policy_path: Fully qualified policy class path (e.g., physicalai.policies.ACT)
            ckpt_path: Path to checkpoint file or export directory (optional)

        Returns:
            Tuple of (policy instance, device string)

        Raises:
            ImportError: If policy class cannot be imported.
            ValueError: If InferenceModel is used without ckpt_path.
        """
        import importlib  # noqa: PLC0415

        from physicalai.devices import get_available_device  # noqa: PLC0415
        from physicalai.inference import InferenceModel  # noqa: PLC0415

        module_path, class_name = policy_path.rsplit(".", 1)
        try:
            policy_class = getattr(importlib.import_module(module_path), class_name)
        except (ImportError, AttributeError) as e:
            msg = f"Could not import policy class '{policy_path}'"
            raise ImportError(msg) from e

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

    def _run_benchmark(self) -> None:
        """Execute benchmark evaluation.

        Raises:
            ValueError: If benchmark configuration is not found.
        """
        from jsonargparse import Namespace  # noqa: PLC0415

        if self.subcommand is None:
            msg = "No subcommand specified"
            raise ValueError(msg)
        config = self.config.get(self.subcommand)
        if config is None:
            msg = "Benchmark configuration not found"
            raise ValueError(msg)

        benchmark_parser = self._subcommand_parsers["benchmark"]
        benchmark = benchmark_parser.instantiate_classes(
            Namespace(benchmark=config.benchmark),
        ).benchmark

        policy, device = self._load_policy(config.policy, config.ckpt_path)

        logger.info("Benchmark: %s", benchmark)
        logger.info("Policy: %s", type(policy).__name__)
        logger.info("Device: %s", device)

        try:
            results = benchmark.evaluate(policy=policy)

            print(results.summary())  # noqa: T201

            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            results.to_json(output_dir / "results.json")
            results.to_csv(output_dir / "results.csv")
            logger.info("Results saved to %s", output_dir)
        finally:
            for gym in benchmark.gyms:
                gym.close()


def cli() -> None:
    """Entry point for physicalai CLI.

    Creates and runs CLI with default configuration for
    physicalai policies, data modules, and trainer.
    """
    CLI(
        model_class=Policy,
        datamodule_class=DataModule,
        trainer_class=Trainer,
        save_config_callback=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    cli()
