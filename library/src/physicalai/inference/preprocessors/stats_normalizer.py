# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stats-based input normalizer using safetensors statistics files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002

from physicalai.inference.preprocessors.base import Preprocessor

_MODE_ALIASES: dict[str, str] = {
    "mean_std": "mean_std",
    "standard": "mean_std",
    "min_max": "min_max",
    "identity": "identity",
}


def _load_stats(path: Path) -> dict[str, dict[str, NDArray[np.floating]]]:
    try:
        from safetensors.numpy import load_file  # noqa: PLC0415
    except ImportError as e:
        msg = "safetensors is required. Install with: pip install safetensors"
        raise ImportError(msg) from e

    flat = load_file(str(path))
    stats: dict[str, dict[str, NDArray[np.floating]]] = {}
    for flat_key, tensor in flat.items():
        feature_name, stat_name = flat_key.rsplit(".", 1)
        if feature_name not in stats:
            stats[feature_name] = {}
        stats[feature_name][stat_name] = tensor.astype(np.float32)
    return stats


class StatsNormalizer(Preprocessor):
    """Normalizes input features using dataset statistics from safetensors."""

    def __init__(
        self,
        mode: str = "mean_std",
        stats_path: str | Path | None = None,
        features: list[str] | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Initialize stats-based input normalizer."""
        self._mode = _MODE_ALIASES.get(mode, mode)
        self._features = set(features or [])
        self._eps = eps
        self._stats: dict[str, dict[str, NDArray[np.floating]]] = {}
        if stats_path is not None:
            self._stats = _load_stats(Path(stats_path))

    def load_stats(self, path: Path) -> None:
        """Load safetensors statistics file."""
        self._stats = _load_stats(path)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Normalize configured input features using loaded stats.

        Returns:
            Input dict with configured features normalized.
        """
        result = dict(inputs)
        for key in self._features:
            if key in result and key in self._stats:
                result[key] = self._transform(result[key], key, inverse=False)
        return result

    def _transform(self, tensor: NDArray[np.floating], key: str, *, inverse: bool) -> NDArray[np.floating]:  # noqa: PLR0911
        if self._mode == "identity" or key not in self._stats:
            return tensor
        stats = self._stats[key]
        if self._mode == "mean_std":
            mean, std = stats.get("mean"), stats.get("std")
            if mean is None or std is None:
                return tensor
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / (std + self._eps)
        if self._mode == "min_max":
            min_val, max_val = stats.get("min"), stats.get("max")
            if min_val is None or max_val is None:
                return tensor
            denom = max_val - min_val
            denom = np.where(denom == 0, self._eps, denom)
            if inverse:
                return (tensor + 1) / 2 * denom + min_val
            return 2 * (tensor - min_val) / denom - 1
        return tensor
