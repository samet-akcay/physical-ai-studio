# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""LeRobot <-> physicalai-studio configuration adapter.

Converts LeRobot's ``TrainPipelineConfig`` (draccus dataclass) into
physicalai-studio's jsonargparse YAML format so that a LeRobot user can
bring their existing training config and run it through the physicalai
``Trainer`` with zero code changes.

Design decisions
----------------
* **Thinnest wrapper** -- we only translate at the config boundary.
  Optimizer / scheduler settings are embedded inside the policy config
  (``LeRobotPolicy.configure_optimizers`` reads from its ``_config``),
  so we do NOT create separate YAML optimizer/scheduler sections.
* **Round-trip safe** -- the generated YAML is valid input for
  ``physicalai fit --config <generated>.yaml``.
* **Lossless where possible** -- fields that have no physicalai
  equivalent are stored under ``_lerobot_extra`` in the YAML so the
  user can inspect them.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning_utilities import module_available

if TYPE_CHECKING or module_available("lerobot"):
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig

    LEROBOT_AVAILABLE = True
else:
    TrainPipelineConfig = None
    DatasetConfig = None
    WandBConfig = None
    LEROBOT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Keys that identify a physicalai jsonargparse config.
_NATIVE_MARKER_KEYS = {"model", "data", "trainer"}

# Keys that identify a LeRobot TrainPipelineConfig.
_LEROBOT_MARKER_KEYS = {"policy", "dataset"}

_DEFAULT_TOLERANCE_S = 1e-4


class _TopLevelMappingTypeError(TypeError, ValueError):
    """Raised when a config does not contain a top-level mapping."""


def detect_config_format(path: str | Path) -> str:
    """Detect whether a config file is native physicalai or LeRobot format.

    The detection is purely structural — it inspects top-level keys:

    * **physicalai**: contains ``model``, ``data``, and/or ``trainer``
      (jsonargparse ``class_path`` / ``init_args`` style).
    * **LeRobot**: contains ``policy`` and ``dataset`` keys
      (draccus dataclass style).

    Supports both YAML (``.yaml``, ``.yml``) and JSON (``.json``) files.

    Args:
        path: Path to the config file.

    Returns:
        ``"physicalai"`` or ``"lerobot"``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the format cannot be determined.
    """
    path = Path(path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    top_keys = _read_top_level_keys(path)

    # Check physicalai first — if it has class_path/init_args structure it's ours
    if top_keys & _NATIVE_MARKER_KEYS:
        return "physicalai"

    if top_keys >= _LEROBOT_MARKER_KEYS:
        return "lerobot"

    msg = (
        f"Cannot determine config format for {path}. "
        f"Top-level keys: {sorted(top_keys)}. "
        f"Expected physicalai keys {sorted(_NATIVE_MARKER_KEYS)} or "
        f"LeRobot keys {sorted(_LEROBOT_MARKER_KEYS)}."
    )
    raise ValueError(msg)


def _read_top_level_keys(path: Path) -> set[str]:
    """Read top-level keys from a YAML or JSON file.

    Returns:
        Set of keys from the top-level mapping.

    Raises:
        ImportError: If YAML parsing is required but PyYAML is unavailable.
        _TopLevelMappingTypeError: If the parsed top-level object is not a mapping.
    """
    suffix = path.suffix.lower()

    if suffix == ".json":
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            msg = "PyYAML is required to read YAML configs. Install with: pip install pyyaml"
            raise ImportError(msg) from e
        with Path(path).open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        # Try YAML first, then JSON as fallback
        try:
            import yaml  # noqa: PLC0415

            with Path(path).open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:  # noqa: BLE001
            with Path(path).open(encoding="utf-8") as f:
                data = json.load(f)

    if not isinstance(data, dict):
        msg = f"Config file {path} does not contain a top-level mapping."
        raise _TopLevelMappingTypeError(msg)

    return set(data.keys())


class TrainPipelineConfigAdapter:
    """Converts a LeRobot ``TrainPipelineConfig`` into a physicalai YAML dict.

    The resulting dict has three top-level keys (``model``, ``data``,
    ``trainer``) that map 1-to-1 to the sections expected by
    ``physicalai fit --config``.

    Usage::

        from lerobot.configs.train import TrainPipelineConfig
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        lr_config = TrainPipelineConfig(...)
        adapter = TrainPipelineConfigAdapter(lr_config)

        # Get as dict
        yaml_dict = adapter.to_dict()

        # Write YAML file
        adapter.to_yaml("converted_config.yaml")
    """

    def __init__(self, config: TrainPipelineConfig) -> None:
        """Initialize adapter with a LeRobot training pipeline config.

        Args:
            config: Source LeRobot ``TrainPipelineConfig`` instance.

        Raises:
            ImportError: If LeRobot is not available in the environment.
        """
        if not LEROBOT_AVAILABLE:
            msg = "TrainPipelineConfigAdapter requires LeRobot.\nInstall with: pip install lerobot"
            raise ImportError(msg)
        self._config = config

    def to_dict(self) -> dict[str, Any]:
        """Convert the LeRobot config to a physicalai-compatible dict.

        Returns:
            Dict with ``model``, ``data``, ``trainer``, and optionally
            ``_lerobot_extra`` keys.
        """
        result: dict[str, Any] = {
            "model": self._map_model(),
            "data": self._map_data(),
            "trainer": self._map_trainer(),
        }

        if self._config.seed is not None:
            result["seed_everything"] = self._config.seed

        extra = self._collect_extra()
        if extra:
            result["_lerobot_extra"] = extra

        return result

    def to_yaml(self, path: str | Path) -> Path:
        """Write the converted config as a YAML file.

        Returns:
            Resolved ``Path`` that was written.

        Raises:
            ImportError: If PyYAML is not available.
        """
        try:
            import yaml  # noqa: PLC0415
        except ImportError as e:
            msg = "PyYAML is required for YAML output. Install with: pip install pyyaml"
            raise ImportError(msg) from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info("Wrote physicalai config to %s", path)
        return path.resolve()

    def _map_model(self) -> dict[str, Any]:
        from physicalai.policies.lerobot import POLICY_CLASS_MAP  # noqa: PLC0415

        cfg = self._config
        policy_cfg = cfg.policy

        if policy_cfg is None:
            msg = "TrainPipelineConfig.policy is None -- cannot determine model class."
            raise ValueError(msg)

        policy_type = getattr(policy_cfg, "type", None)
        if policy_type is None:
            msg = "Cannot determine policy type from config.policy."
            raise ValueError(msg)

        class_path = POLICY_CLASS_MAP.get(policy_type)
        if class_path is None:
            class_path = "physicalai.policies.lerobot.LeRobotPolicy"
            logger.warning(
                "No explicit wrapper for policy type %r -- falling back to LeRobotPolicy universal wrapper.",
                policy_type,
            )

        init_args: dict[str, Any] = {"policy_name": policy_type}

        policy_config_dict = self._policy_config_to_dict(policy_cfg)
        if policy_config_dict:
            init_args["policy_config"] = policy_config_dict

        return {"class_path": class_path, "init_args": init_args}

    def _map_data(self) -> dict[str, Any]:
        cfg = self._config
        ds = cfg.dataset

        init_args: dict[str, Any] = {
            "repo_id": ds.repo_id,
            "train_batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "data_format": "lerobot",
        }

        if ds.root is not None:
            init_args["root"] = str(ds.root)
        if ds.episodes is not None:
            init_args["episodes"] = ds.episodes
        if ds.revision is not None:
            init_args["revision"] = ds.revision
        if ds.video_backend:
            init_args["video_backend"] = ds.video_backend
        if cfg.tolerance_s != _DEFAULT_TOLERANCE_S:
            init_args["tolerance_s"] = cfg.tolerance_s

        return {
            "class_path": "physicalai.data.lerobot.LeRobotDataModule",
            "init_args": init_args,
        }

    def _map_trainer(self) -> dict[str, Any]:
        cfg = self._config
        trainer: dict[str, Any] = {
            "max_steps": cfg.steps,
            "log_every_n_steps": cfg.log_freq,
            "val_check_interval": cfg.eval_freq,
            "enable_checkpointing": cfg.save_checkpoint,
            "accelerator": "auto",
            "devices": "auto",
        }

        if cfg.output_dir is not None:
            trainer["default_root_dir"] = str(cfg.output_dir)

        grad_clip = self._extract_grad_clip_norm()
        if grad_clip is not None:
            trainer["gradient_clip_val"] = grad_clip

        use_amp = getattr(cfg.policy, "use_amp", False) if cfg.policy else False
        trainer["precision"] = "16-mixed" if use_amp else "32"

        if getattr(cfg, "cudnn_deterministic", False):
            trainer["deterministic"] = True

        callbacks = self._build_callbacks(cfg)
        if callbacks:
            trainer["callbacks"] = callbacks

        loggers = self._build_loggers(cfg)
        if loggers:
            trainer["logger"] = loggers

        return trainer

    @staticmethod
    def _build_callbacks(cfg: TrainPipelineConfig) -> list[dict[str, Any]]:
        callbacks: list[dict[str, Any]] = []

        if cfg.save_checkpoint and cfg.save_freq > 0:
            callbacks.append({
                "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                "init_args": {
                    "every_n_train_steps": cfg.save_freq,
                    "save_top_k": -1,
                },
            })

        return callbacks

    @staticmethod
    def _build_loggers(cfg: TrainPipelineConfig) -> list[dict[str, Any]]:
        loggers: list[dict[str, Any]] = []

        wandb = cfg.wandb
        if wandb.enable:
            init_args: dict[str, Any] = {
                "project": wandb.project,
            }
            if wandb.entity is not None:
                init_args["entity"] = wandb.entity
            if wandb.notes is not None:
                init_args["notes"] = wandb.notes
            if wandb.run_id is not None:
                init_args["id"] = wandb.run_id
            if wandb.mode is not None:
                init_args["mode"] = wandb.mode
            if cfg.job_name:
                init_args["name"] = cfg.job_name

            loggers.append({
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": init_args,
            })

        return loggers

    def _extract_grad_clip_norm(self) -> float | None:
        optimizer = self._config.optimizer
        if optimizer is not None and hasattr(optimizer, "grad_clip_norm"):
            return optimizer.grad_clip_norm

        policy = self._config.policy
        if policy is not None and hasattr(policy, "get_optimizer_preset"):
            try:
                preset = policy.get_optimizer_preset()
                if hasattr(preset, "grad_clip_norm"):
                    return preset.grad_clip_norm
            except Exception:  # noqa: BLE001, S110
                pass

        return None

    def _collect_extra(self) -> dict[str, Any]:
        cfg = self._config
        extra: dict[str, Any] = {}

        if cfg.resume:
            extra["resume"] = True
        if cfg.use_rabc:
            extra["rabc"] = {
                "enabled": True,
                "progress_path": cfg.rabc_progress_path,
                "kappa": cfg.rabc_kappa,
                "epsilon": cfg.rabc_epsilon,
                "head_mode": cfg.rabc_head_mode,
            }
        if cfg.peft is not None:
            extra["peft"] = dataclasses.asdict(cfg.peft)
        if cfg.env is not None:
            extra["env"] = str(cfg.env)
        if cfg.rename_map:
            extra["rename_map"] = cfg.rename_map

        return extra

    @staticmethod
    def _policy_config_to_dict(policy_cfg: Any) -> dict[str, Any]:  # noqa: ANN401
        """Extract serialisable policy config params.

        Skips internal fields (``input_features``, ``output_features``)
        that are populated at runtime by the dataset, and also skips
        ``type`` which is used for class dispatch.

        Returns:
            Dictionary of serialisable non-default policy config values.
        """
        skip = {"input_features", "output_features", "type", "pretrained_path"}
        result: dict[str, Any] = {}

        if not dataclasses.is_dataclass(policy_cfg):
            return result

        for f in dataclasses.fields(policy_cfg):
            if f.name in skip:
                continue
            val = getattr(policy_cfg, f.name)
            if val == f.default:
                continue
            if callable(f.default_factory) and val == f.default_factory():
                continue
            result[f.name] = _serialize_value(val)

        return result

    @classmethod
    def from_file(cls, path: str | Path) -> TrainPipelineConfigAdapter:
        """Load a ``TrainPipelineConfig`` from a JSON file and wrap it.

        This is a convenience constructor for users who have an existing
        ``train_config.json`` (e.g. from a LeRobot training run).

        Returns:
            Adapter instance wrapping the loaded config.

        Raises:
            ImportError: If LeRobot is not available.
        """
        if not LEROBOT_AVAILABLE:
            msg = "LeRobot is required. Install with: pip install lerobot"
            raise ImportError(msg)

        from lerobot.configs.train import TrainPipelineConfig  # noqa: PLC0415

        config = TrainPipelineConfig.from_pretrained(str(path))
        return cls(config)


def _serialize_value(val: Any) -> Any:  # noqa: ANN401
    """Recursively serialize a value for YAML output.

    Returns:
        YAML-serializable representation of ``val``.
    """
    if isinstance(val, Path):
        return str(val)
    if dataclasses.is_dataclass(val) and not isinstance(val, type):
        return dataclasses.asdict(val)
    if isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    return str(val)


__all__ = ["TrainPipelineConfigAdapter", "detect_config_format"]
