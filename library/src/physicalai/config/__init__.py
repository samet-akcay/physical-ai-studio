# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Configuration utilities for physicalai."""

from physicalai.config.base import Config
from physicalai.config.instantiate import instantiate_obj
from physicalai.config.mixin import FromConfig

__all__ = ["Config", "FromConfig", "TrainPipelineConfigAdapter", "detect_config_format", "instantiate_obj"]  # noqa: F822


def __getattr__(name: str) -> object:
    if name in {"TrainPipelineConfigAdapter", "detect_config_format"}:
        from physicalai.config import lerobot as _lr  # noqa: PLC0415

        return getattr(_lr, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
