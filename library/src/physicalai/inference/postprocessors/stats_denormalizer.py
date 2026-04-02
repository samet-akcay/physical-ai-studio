# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stats-based output denormalizer using safetensors statistics files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002

from physicalai.inference.postprocessors.base import Postprocessor

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


class StatsDenormalizer(Postprocessor):
    """Denormalizes output features using dataset statistics from safetensors."""

    def __init__(
        self,
        mode: str = "mean_std",
        stats_path: str | Path | None = None,
        features: list[str] | None = None,
        eps: float = 1e-8,
    ) -> None:
        """Initialize stats-based output denormalizer."""
        self._mode = _MODE_ALIASES.get(mode, mode)
        self._features = set(features or [])
        self._eps = eps
        self._stats: dict[str, dict[str, NDArray[np.floating]]] = {}
        if stats_path is not None:
            self._stats = _load_stats(Path(stats_path))

    def load_stats(self, path: Path) -> None:
        """Load safetensors statistics file."""
        self._stats = _load_stats(path)

    def __call__(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Denormalize configured output features using loaded stats.

        Returns:
            Output dict with configured features denormalized.
        """
        result = dict(outputs)
        for key in self._features:
            if key in result and key in self._stats:
                result[key] = self._transform(result[key], key)
        return result

    def _transform(self, tensor: NDArray[np.floating], key: str) -> NDArray[np.floating]:
        if self._mode == "identity" or key not in self._stats:
            return tensor
        stats = self._stats[key]
        if self._mode == "mean_std":
            mean, std = stats.get("mean"), stats.get("std")
            if mean is None or std is None:
                return tensor
            return tensor * std + mean
        if self._mode == "min_max":
            min_val, max_val = stats.get("min"), stats.get("max")
            if min_val is None or max_val is None:
                return tensor
            denom = max_val - min_val
            denom = np.where(denom == 0, self._eps, denom)
            return (tensor + 1) / 2 * denom + min_val
        return tensor
