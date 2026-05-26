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
    """LeRobot MolmoAct2 policy.

    Use the standard LeRobot wrapper flows:

    * ``MolmoAct2(config=MolmoAct2Config(...))`` for direct construction,
    * ``MolmoAct2.from_pretrained("allenai/MolmoAct2-SO100_101")`` for a
      released checkpoint,
    * ``MolmoAct2.load_from_checkpoint(path)`` for a Lightning checkpoint.

    Importing this module registers ``MolmoAct2Config`` with LeRobot's
    config registry; the wrapper itself adds nothing beyond binding
    ``POLICY_NAME`` so ``isinstance(policy, MolmoAct2)`` discriminates.
    """

    POLICY_NAME = "molmoact2"
