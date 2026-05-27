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

    Production use (robot / :class:`~physicalai.runtime.PolicyRuntime`) should
    use ``MolmoAct2.from_config(MolmoAct2Config(...))`` then ``export`` and
    :class:`~physicalai.inference.InferenceModel.load` — see
    ``library/scripts/molmoact2_dry_run.py``.

    Supported construction paths:

    * ``MolmoAct2.from_config(MolmoAct2Config(...))`` — deployment and fine-tuning
      (build the LeRobot config with explicit fields, then pass it here),
    * ``MolmoAct2.from_pretrained("allenai/MolmoAct2-SO100_101")`` — Hub checkpoint,
    * ``MolmoAct2.load_from_checkpoint(path)`` — torch export / Lightning checkpoint,
    * ``MolmoAct2.from_dataset(...)`` — generic LeRobot scaffold from dataset metadata.

    Importing this module registers ``MolmoAct2Config`` with LeRobot's config
    registry. The wrapper only binds ``POLICY_NAME`` for type discrimination.
    """

    POLICY_NAME = "molmoact2"
