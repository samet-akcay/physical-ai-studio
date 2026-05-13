# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for optional MolmoAct2 integration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from physicalai.policies import MolmoAct2, get_policy


class DummyMolmoAct2Config:
    def __init__(self, **kwargs):  # noqa: ANN003
        self.kwargs = kwargs
        self.checkpoint_path = kwargs.get("checkpoint_path", "")
        self.norm_tag = kwargs.get("norm_tag", "")


class DummyMolmoAct2Policy(torch.nn.Module):
    def __init__(self, config: DummyMolmoAct2Config) -> None:
        super().__init__()
        self.config = config


def _identity_processor(batch):  # noqa: ANN001, ANN202
    return batch


def test_factory_returns_molmoact2_wrapper() -> None:
    with pytest.warns(UserWarning, match="not in physicalai's supported set"):
        policy = get_policy("molmoact2")

    assert isinstance(policy, MolmoAct2)
    assert policy.policy_name == "molmoact2"


def test_from_checkpoint_builds_fork_config() -> None:
    with (
        patch(
        "physicalai.policies.molmoact2.policy._get_molmoact2_config_class",
        return_value=DummyMolmoAct2Config,
        ),
        patch("physicalai.policies.lerobot.policy.get_policy_class", return_value=DummyMolmoAct2Policy),
        patch(
            "physicalai.policies.lerobot.policy.make_pre_post_processors",
            return_value=(_identity_processor, _identity_processor),
        ),
    ):
        policy = MolmoAct2.from_checkpoint(
            "allenai/MolmoAct2-LIBERO",
            norm_tag="libero",
            num_steps=8,
            enable_adaptive_depth=False,
        )

    assert policy.policy_name == "molmoact2"
    assert policy._config.checkpoint_path == "allenai/MolmoAct2-LIBERO"
    assert policy._config.norm_tag == "libero"
    assert policy._config.kwargs["num_steps"] == 8
    assert policy._config.kwargs["enable_adaptive_depth"] is False


def test_missing_molmoact2_fork_error_is_actionable() -> None:
    with patch(
        "physicalai.policies.molmoact2.policy._get_molmoact2_config_class",
        side_effect=ImportError("install fork"),
    ):
        with pytest.raises(ImportError, match="install fork"):
            MolmoAct2.from_checkpoint("allenai/MolmoAct2-LIBERO", norm_tag="libero")
