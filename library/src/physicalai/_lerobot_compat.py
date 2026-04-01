# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: INP001
"""Compatibility shim for lerobot 0.5.0 vs 0.5.1 import differences.

In lerobot 0.5.1, several functions moved from ``lerobot.datasets.utils``
to ``lerobot.datasets.feature_utils``. This module re-exports them from
whichever location is available so the rest of the codebase can use a
single, stable import path.
"""

from __future__ import annotations

try:
    from lerobot.datasets.feature_utils import (  # lerobot >=0.5.1
        build_dataset_frame,
        check_delta_timestamps,
        combine_feature_dicts,
        dataset_to_policy_features,
        get_delta_indices,
    )
except ImportError:
    from lerobot.datasets.utils import (  # lerobot 0.5.0
        build_dataset_frame,
        check_delta_timestamps,
        combine_feature_dicts,
        dataset_to_policy_features,
        get_delta_indices,
    )

__all__ = [
    "build_dataset_frame",
    "check_delta_timestamps",
    "combine_feature_dicts",
    "dataset_to_policy_features",
    "get_delta_indices",
]
