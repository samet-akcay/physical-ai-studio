# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the LeRobot MolmoAct2 wrapper."""

# ruff: noqa: S101

from __future__ import annotations

from unittest.mock import patch

import torch

from physicalai.policies.lerobot import MolmoAct2, get_lerobot_policy
from physicalai.policies.lerobot.policy import NamedLeRobotPolicy


class DummyMolmoAct2Config:
    """Minimal stand-in for LeRobot's MolmoAct2Config."""

    type = "molmoact2"

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.checkpoint_path = kwargs.get("checkpoint_path", "")
        self.norm_tag = kwargs.get("norm_tag")

    def get_optimizer_preset(self):  # noqa: ANN201, PLR6301
        class _Preset:
            lr = 1e-5

        return _Preset()


class DummyMolmoAct2Policy(torch.nn.Module):
    """Minimal stand-in for LeRobot's MolmoAct2Policy."""

    def __init__(self, config: DummyMolmoAct2Config) -> None:
        super().__init__()
        self.config = config


def _identity_processor(batch):  # noqa: ANN001, ANN202
    return batch


def test_molmoact2_is_named_lerobot_policy() -> None:
    """MolmoAct2 is the standard 3-line NamedLeRobotPolicy shape."""
    assert issubclass(MolmoAct2, NamedLeRobotPolicy)
    assert MolmoAct2.POLICY_NAME == "molmoact2"


def test_get_lerobot_policy_returns_molmoact2_wrapper() -> None:
    """MolmoAct2 is reachable via the generic get_lerobot_policy factory."""
    with (
        patch("physicalai.policies.lerobot.LEROBOT_AVAILABLE", new=True),
        patch("physicalai.policies.lerobot.policy.LEROBOT_AVAILABLE", new=True),
    ):
        policy = get_lerobot_policy("molmoact2")

    assert type(policy) is MolmoAct2
    assert policy.policy_name == "molmoact2"
    assert policy.config is None


def test_direct_construction_with_config_initializes_wrapper() -> None:
    """MolmoAct2(config=...) goes through the standard NamedLeRobotPolicy path."""
    with (
        patch("physicalai.policies.lerobot.policy.get_policy_class", return_value=DummyMolmoAct2Policy),
        patch(
            "physicalai.policies.lerobot.policy.make_pre_post_processors",
            return_value=(_identity_processor, _identity_processor),
        ),
        patch("physicalai.policies.lerobot.policy.LEROBOT_AVAILABLE", new=True),
    ):
        config = DummyMolmoAct2Config(
            checkpoint_path="allenai/MolmoAct2-SO100_101",
            norm_tag="so100_so101_molmoact2",
        )
        policy = MolmoAct2(config=config)

    assert policy.policy_name == "molmoact2"
    assert policy.config is config
    assert isinstance(policy.lerobot_policy, DummyMolmoAct2Policy)
