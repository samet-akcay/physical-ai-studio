# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2 LeRobot policy wrapper."""

from __future__ import annotations

import importlib

from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

try:
    importlib.import_module("lerobot.policies.molmoact2.configuration_molmoact2")
except ImportError:
    pass


class MolmoAct2(NamedLeRobotPolicy):
    POLICY_NAME = "molmoact2"
