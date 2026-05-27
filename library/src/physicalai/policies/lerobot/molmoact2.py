# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 LeRobot policy wrapper."""

from __future__ import annotations

import importlib
from contextlib import suppress

from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

with suppress(ImportError):
    importlib.import_module("lerobot.policies.molmoact2.configuration_molmoact2")


class MolmoAct2(NamedLeRobotPolicy):
    """Named LeRobot wrapper for MolmoAct2."""

    POLICY_NAME = "molmoact2"
