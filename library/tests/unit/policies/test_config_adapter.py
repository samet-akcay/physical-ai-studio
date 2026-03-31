# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for LeRobot configuration adapter behavior."""

# ruff: noqa: S101, PLR6301, PLC0415, D102
# ruff: noqa: D101, D103, ANN001, ANN201, S108, PLR2004

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from lerobot.configs.default import DatasetConfig, WandBConfig

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("lerobot", reason="LeRobot not installed"),
    reason="Requires lerobot",
)


def _make_policy_mock(policy_type: str = "act", *, use_amp: bool = False) -> MagicMock:
    policy_cfg = MagicMock(spec=[])
    policy_cfg.type = policy_type
    policy_cfg.use_amp = use_amp
    return policy_cfg


def _make_train_config(*, policy_type: str = "act", wandb_enable: bool = False) -> MagicMock:
    config = MagicMock()
    config.policy = _make_policy_mock(policy_type)
    config.dataset = DatasetConfig(repo_id="lerobot/pusht")
    config.env = None
    config.output_dir = Path("/tmp/test_output")
    config.job_name = "test_job"
    config.resume = False
    config.seed = 42
    config.cudnn_deterministic = False
    config.num_workers = 4
    config.batch_size = 32
    config.steps = 50000
    config.eval_freq = 10000
    config.log_freq = 100
    config.tolerance_s = 1e-4
    config.save_checkpoint = True
    config.save_freq = 5000
    config.optimizer = None
    config.scheduler = None
    config.wandb = (
        WandBConfig(enable=wandb_enable)
        if not wandb_enable
        else MagicMock(
            enable=True,
            project="test_project",
            entity="test_entity",
            notes=None,
            run_id=None,
            mode=None,
        )
    )
    config.peft = None
    config.use_rabc = False
    config.rabc_progress_path = None
    config.rabc_kappa = 0.01
    config.rabc_epsilon = 1e-6
    config.rabc_head_mode = "sparse"
    config.rename_map = {}
    return config


@pytest.fixture(scope="module")
def lerobot_train_config():
    return _make_train_config()


class TestTrainPipelineConfigAdapterToDict:
    def test_top_level_keys(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()

        assert "model" in result
        assert "data" in result
        assert "trainer" in result

    def test_model_class_path_maps_act(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()

        assert result["model"]["class_path"] == "physicalai.policies.lerobot.ACT"
        assert result["model"]["init_args"]["policy_name"] == "act"

    def test_data_section(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()
        data = result["data"]

        assert data["class_path"] == "physicalai.data.lerobot.LeRobotDataModule"
        assert data["init_args"]["repo_id"] == "lerobot/pusht"
        assert data["init_args"]["train_batch_size"] == 32
        assert data["init_args"]["num_workers"] == 4
        assert data["init_args"]["data_format"] == "lerobot"

    def test_trainer_section(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()
        trainer = result["trainer"]

        assert trainer["max_steps"] == 50000
        assert trainer["log_every_n_steps"] == 100
        assert trainer["val_check_interval"] == 10000
        assert trainer["enable_checkpointing"] is True
        assert trainer["default_root_dir"] == "/tmp/test_output"

    def test_precision_32_when_no_amp(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()

        assert result["trainer"]["precision"] == "32"

    def test_precision_16_mixed_when_amp(self):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config()
        config.policy.use_amp = True
        adapter = TrainPipelineConfigAdapter(config)
        result = adapter.to_dict()
        assert result["trainer"]["precision"] == "16-mixed"

    def test_checkpoint_callback_created(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()
        callbacks = result["trainer"]["callbacks"]

        assert len(callbacks) == 1
        assert callbacks[0]["class_path"] == "lightning.pytorch.callbacks.ModelCheckpoint"
        assert callbacks[0]["init_args"]["every_n_train_steps"] == 5000

    def test_wandb_logger_when_enabled(self):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config(wandb_enable=True)
        adapter = TrainPipelineConfigAdapter(config)
        result = adapter.to_dict()
        loggers = result["trainer"]["logger"]

        assert len(loggers) == 1
        assert loggers[0]["class_path"] == "lightning.pytorch.loggers.WandbLogger"
        assert loggers[0]["init_args"]["project"] == "test_project"
        assert loggers[0]["init_args"]["entity"] == "test_entity"

    def test_no_logger_when_wandb_disabled(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()

        assert "logger" not in result["trainer"]

    def test_seed_maps_to_seed_everything(self, lerobot_train_config):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        result = adapter.to_dict()

        assert result["seed_everything"] == 42

    def test_grad_clip_from_optimizer(self):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config()
        config.optimizer = MagicMock(grad_clip_norm=5.0)
        adapter = TrainPipelineConfigAdapter(config)
        result = adapter.to_dict()
        assert result["trainer"]["gradient_clip_val"] == 5.0

    def test_unknown_policy_type_falls_back_to_universal(self):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config(policy_type="some_future_policy")
        adapter = TrainPipelineConfigAdapter(config)
        result = adapter.to_dict()
        assert result["model"]["class_path"] == "physicalai.policies.lerobot.LeRobotPolicy"

    def test_raises_when_policy_is_none(self):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config()
        config.policy = None

        adapter = TrainPipelineConfigAdapter(config)
        with pytest.raises(ValueError, match="policy is None"):
            adapter.to_dict()


class TestTrainPipelineConfigAdapterToYaml:
    def test_writes_valid_yaml(self, lerobot_train_config, tmp_path):
        import yaml

        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        output = tmp_path / "test_config.yaml"
        result_path = adapter.to_yaml(output)

        assert result_path.exists()

        with Path(result_path).open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

        assert "model" in loaded
        assert "data" in loaded
        assert "trainer" in loaded
        assert loaded["trainer"]["max_steps"] == 50000

    def test_creates_parent_dirs(self, lerobot_train_config, tmp_path):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        adapter = TrainPipelineConfigAdapter(lerobot_train_config)
        output = tmp_path / "nested" / "dir" / "config.yaml"
        result_path = adapter.to_yaml(output)

        assert result_path.exists()


class TestTrainPipelineConfigAdapterPolicyMapping:
    @pytest.mark.parametrize(
        ("policy_type", "expected_class_path"),
        [
            ("act", "physicalai.policies.lerobot.ACT"),
            ("diffusion", "physicalai.policies.lerobot.Diffusion"),
            ("smolvla", "physicalai.policies.lerobot.SmolVLA"),
            ("groot", "physicalai.policies.lerobot.Groot"),
            ("vqbet", "physicalai.policies.lerobot.VQBeT"),
            ("tdmpc", "physicalai.policies.lerobot.TDMPC"),
            ("sac", "physicalai.policies.lerobot.SAC"),
            ("pi0", "physicalai.policies.lerobot.PI0"),
            ("pi05", "physicalai.policies.lerobot.PI05"),
            ("pi0_fast", "physicalai.policies.lerobot.PI0Fast"),
        ],
    )
    def test_policy_type_to_class_path(self, policy_type, expected_class_path):
        from physicalai.config.lerobot import TrainPipelineConfigAdapter

        config = _make_train_config(policy_type=policy_type)
        adapter = TrainPipelineConfigAdapter(config)
        result = adapter.to_dict()
        assert result["model"]["class_path"] == expected_class_path
