# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for detect_config_format(), CLI config subcommand, and auto-detection."""

# ruff: noqa: S101, PLR6301, PLC0415, PLC2701, SLF001, D102, DOC201

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import yaml

if TYPE_CHECKING:
    import argparse

    from physicalai.cli.cli import CLI

# ───────────────────────────────────────────────────────────────────
# detect_config_format
# ───────────────────────────────────────────────────────────────────


class TestDetectConfigFormat:
    """Unit tests for ``physicalai.config.lerobot.detect_config_format``."""

    # -- Native configs ------------------------------------------------

    def test_native_yaml_with_model_data_trainer(self, tmp_path: Path) -> None:
        """YAML with model/data/trainer keys → 'native'."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "native.yaml"
        cfg.write_text(yaml.dump({"model": {}, "data": {}, "trainer": {}}))
        assert detect_config_format(cfg) == "physicalai"

    def test_native_json_with_model_and_trainer(self, tmp_path: Path) -> None:
        """JSON with model+trainer (no data) → still 'native' (any marker key suffices)."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "native.json"
        cfg.write_text(json.dumps({"model": {"class_path": "Foo"}, "trainer": {"max_epochs": 10}}))
        assert detect_config_format(cfg) == "physicalai"

    def test_native_with_only_data_key(self, tmp_path: Path) -> None:
        """YAML with only 'data' → 'native' (single native marker key is enough)."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "data_only.yaml"
        cfg.write_text(yaml.dump({"data": {"class_path": "MyData"}}))
        assert detect_config_format(cfg) == "physicalai"

    def test_native_with_only_trainer_key(self, tmp_path: Path) -> None:
        """YAML with only 'trainer' → 'native'."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "trainer_only.yaml"
        cfg.write_text(yaml.dump({"trainer": {"max_epochs": 5}}))
        assert detect_config_format(cfg) == "physicalai"

    # -- LeRobot configs -----------------------------------------------

    def test_lerobot_json_with_policy_and_dataset(self, tmp_path: Path) -> None:
        """JSON with policy+dataset keys → 'lerobot'."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "lerobot.json"
        cfg.write_text(json.dumps({"policy": {"type": "act"}, "dataset": {"repo_id": "x/y"}}))
        assert detect_config_format(cfg) == "lerobot"

    def test_lerobot_yaml_with_policy_dataset_and_extras(self, tmp_path: Path) -> None:
        """YAML with policy+dataset + extra LeRobot keys → 'lerobot'."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "lerobot.yaml"
        cfg.write_text(
            yaml.dump({
                "policy": {"type": "diffusion"},
                "dataset": {"repo_id": "x/y"},
                "env": None,
                "steps": 100000,
            }),
        )
        assert detect_config_format(cfg) == "lerobot"

    def test_lerobot_with_only_policy_not_enough(self, tmp_path: Path) -> None:
        """Only 'policy' without 'dataset' → ValueError (need both markers)."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "partial.json"
        cfg.write_text(json.dumps({"policy": {"type": "act"}, "steps": 100}))
        with pytest.raises(ValueError, match="Cannot determine config format"):
            detect_config_format(cfg)

    # -- Ambiguous / overlapping keys -----------------------------------

    def test_native_wins_when_both_markers_present(self, tmp_path: Path) -> None:
        """Config with BOTH native and LeRobot marker keys → 'native' (checked first)."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "ambiguous.yaml"
        cfg.write_text(
            yaml.dump({
                "model": {},
                "data": {},
                "trainer": {},
                "policy": {},
                "dataset": {},
            }),
        )
        assert detect_config_format(cfg) == "physicalai"

    # -- Error cases ---------------------------------------------------

    def test_file_not_found_raises(self) -> None:
        """Non-existent file → FileNotFoundError."""
        from physicalai.config.lerobot import detect_config_format

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            detect_config_format("/nonexistent/config.yaml")

    def test_unrecognized_keys_raise_value_error(self, tmp_path: Path) -> None:
        """Config with no recognised marker keys → ValueError."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "unknown.json"
        cfg.write_text(json.dumps({"foo": 1, "bar": 2}))
        with pytest.raises(ValueError, match="Cannot determine config format"):
            detect_config_format(cfg)

    def test_non_dict_top_level_raises(self, tmp_path: Path) -> None:
        """Config whose top-level is a list → ValueError."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "list.json"
        cfg.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="does not contain a top-level mapping"):
            detect_config_format(cfg)

    def test_empty_dict_raises(self, tmp_path: Path) -> None:
        """Empty dict → ValueError (no keys match anything)."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "empty.yaml"
        cfg.write_text(yaml.dump({}))
        with pytest.raises(ValueError, match="Cannot determine config format"):
            detect_config_format(cfg)

    # -- File extension handling ----------------------------------------

    def test_yml_extension_works(self, tmp_path: Path) -> None:
        """.yml extension parsed correctly."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "config.yml"
        cfg.write_text(yaml.dump({"model": {}, "trainer": {}}))
        assert detect_config_format(cfg) == "physicalai"

    def test_unknown_extension_falls_back_to_yaml_then_json(self, tmp_path: Path) -> None:
        """File without .json/.yaml/.yml extension — tries YAML parsing first."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "config.txt"
        cfg.write_text(yaml.dump({"policy": {"type": "act"}, "dataset": {"repo_id": "x"}}))
        assert detect_config_format(cfg) == "lerobot"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """String path (not Path object) works fine."""
        from physicalai.config.lerobot import detect_config_format

        cfg = tmp_path / "native.json"
        cfg.write_text(json.dumps({"model": {}}))
        assert detect_config_format(str(cfg)) == "physicalai"


# ───────────────────────────────────────────────────────────────────
# _read_top_level_keys (internal helper)
# ───────────────────────────────────────────────────────────────────


class TestReadTopLevelKeys:
    """Tests for the ``_read_top_level_keys`` internal helper."""

    def test_reads_json_keys(self, tmp_path: Path) -> None:
        from physicalai.config.lerobot import _read_top_level_keys

        cfg = tmp_path / "test.json"
        cfg.write_text(json.dumps({"a": 1, "b": 2, "c": 3}))
        assert _read_top_level_keys(cfg) == {"a", "b", "c"}

    def test_reads_yaml_keys(self, tmp_path: Path) -> None:
        from physicalai.config.lerobot import _read_top_level_keys

        cfg = tmp_path / "test.yaml"
        cfg.write_text(yaml.dump({"x": 10, "y": 20}))
        assert _read_top_level_keys(cfg) == {"x", "y"}

    def test_non_dict_raises(self, tmp_path: Path) -> None:
        from physicalai.config.lerobot import _read_top_level_keys

        cfg = tmp_path / "scalar.yaml"
        cfg.write_text("just a string")
        with pytest.raises(ValueError, match="does not contain a top-level mapping"):
            _read_top_level_keys(cfg)


# ───────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────


class TestMaybeConvertLerobotConfig:
    """Tests for ``CLI._maybe_convert_lerobot_config``.

    We instantiate a bare ``CLI`` object (bypassing ``__init__``) to test
    the method in isolation, mocking the adapter and detect functions.
    """

    @staticmethod
    def _make_cli_stub() -> CLI:
        """Create a minimal CLI-like object with ``_maybe_convert_lerobot_config``."""
        from physicalai.cli.cli import CLI

        obj = object.__new__(CLI)
        obj._temp_config_file = None
        return obj

    def test_lerobot_config_is_swapped(self, tmp_path: Path) -> None:
        """When --config points to a LeRobot file, it's replaced with a temp YAML."""
        cli = self._make_cli_stub()
        lerobot_cfg = tmp_path / "lr_config.json"
        lerobot_cfg.write_text(json.dumps({"policy": {"type": "act"}, "dataset": {"repo_id": "x/y"}}))

        mock_adapter = MagicMock()
        mock_adapter.to_yaml.return_value = tmp_path / "converted.yaml"

        with (
            patch("physicalai.config.lerobot.detect_config_format", return_value="lerobot"),
            patch("physicalai.config.lerobot.TrainPipelineConfigAdapter.from_file", return_value=mock_adapter),
        ):
            args = ["fit", "--config", str(lerobot_cfg)]
            result = cli._maybe_convert_lerobot_config(args)

        config_idx = result.index("--config")
        new_path = result[config_idx + 1]
        assert new_path != str(lerobot_cfg)
        assert new_path.endswith(".yaml")

    def test_native_config_passes_through(self, tmp_path: Path) -> None:
        """When --config points to a native file, args are unchanged."""
        cli = self._make_cli_stub()
        native_cfg = tmp_path / "native.yaml"
        native_cfg.write_text(yaml.dump({"model": {}, "data": {}, "trainer": {}}))

        args = ["fit", "--config", str(native_cfg)]
        result = cli._maybe_convert_lerobot_config(args)

        config_idx = result.index("--config")
        assert result[config_idx + 1] == str(native_cfg)

    def test_non_training_subcommand_skipped(self) -> None:
        """Non-training subcommands (benchmark, config) are not touched."""
        cli = self._make_cli_stub()

        args = ["benchmark", "--config", "something.json"]
        result = cli._maybe_convert_lerobot_config(args)
        assert result == args

    def test_no_config_flag_returns_unchanged(self) -> None:
        """Args without --config pass through."""
        cli = self._make_cli_stub()

        args = ["fit", "--trainer.max_epochs", "10"]
        result = cli._maybe_convert_lerobot_config(args)
        assert result == args

    def test_nonexistent_config_path_skipped(self) -> None:
        """--config pointing to a missing file is skipped (no crash)."""
        cli = self._make_cli_stub()

        args = ["fit", "--config", "/nonexistent/path.json"]
        result = cli._maybe_convert_lerobot_config(args)
        assert result == args

    def test_none_args_uses_sys_argv(self, tmp_path: Path) -> None:
        """When args is None, reads from sys.argv[1:]."""
        cli = self._make_cli_stub()
        native_cfg = tmp_path / "native.yaml"
        native_cfg.write_text(yaml.dump({"model": {}, "data": {}, "trainer": {}}))

        with patch("sys.argv", ["physicalai", "fit", "--config", str(native_cfg)]):
            result = cli._maybe_convert_lerobot_config(None)

        assert result is None

    def test_empty_args_returns_unchanged(self) -> None:
        """Empty arg list returns as-is."""
        cli = self._make_cli_stub()
        result = cli._maybe_convert_lerobot_config([])
        assert result == []

    def test_detect_raises_value_error_skips_gracefully(self, tmp_path: Path) -> None:
        """If detect_config_format raises ValueError, the config is skipped."""
        cli = self._make_cli_stub()
        cfg = tmp_path / "weird.json"
        cfg.write_text(json.dumps({"foo": 1}))

        args = ["fit", "--config", str(cfg)]
        result = cli._maybe_convert_lerobot_config(args)

        config_idx = result.index("--config")
        assert result[config_idx + 1] == str(cfg)


# ───────────────────────────────────────────────────────────────────
# CLI._create_config_parser
# ───────────────────────────────────────────────────────────────────


class TestCreateConfigParser:
    """Tests for the ``config`` subcommand argument parser."""

    @staticmethod
    def _get_config_parser() -> argparse.ArgumentParser:
        """Get a config parser from a CLI stub."""
        from physicalai.cli.cli import CLI

        cli = object.__new__(CLI)
        cli._temp_config_file = None
        return cli._create_config_parser(description="test")

    def test_parser_has_from_argument(self) -> None:
        parser = self._get_config_parser()
        ns = parser.parse_args(["input.json"])
        assert ns.source_format == "lerobot"

    def test_parser_from_flag(self) -> None:
        parser = self._get_config_parser()
        ns = parser.parse_args(["--from", "lerobot", "input.json"])
        assert ns.source_format == "lerobot"
        assert ns.input == "input.json"

    def test_parser_output_flag(self) -> None:
        parser = self._get_config_parser()
        ns = parser.parse_args(["input.json", "-o", "output.yaml"])
        assert ns.output == "output.yaml"

    def test_parser_output_default_is_none(self) -> None:
        parser = self._get_config_parser()
        ns = parser.parse_args(["input.json"])
        assert ns.output is None


# ───────────────────────────────────────────────────────────────────
# CLI._run_config
# ───────────────────────────────────────────────────────────────────


class TestRunConfig:
    """Tests for ``CLI._run_config`` — the explicit config conversion path."""

    @staticmethod
    def _make_cli_stub_with_config(
        source_format: str,
        input_path: str,
        output_path: str | None = None,
    ) -> CLI:
        """Create a CLI stub with a mocked ``self.config`` namespace."""
        from physicalai.cli.cli import CLI

        cli = object.__new__(CLI)
        cli._temp_config_file = None
        cli.subcommand = "config"

        config_ns = MagicMock()
        config_ns.source_format = source_format
        config_ns.input = input_path
        config_ns.output = output_path

        cli.config = MagicMock()
        cli.config.get.return_value = config_ns
        return cli

    def test_run_config_calls_adapter(self, tmp_path: Path) -> None:
        """_run_config calls TrainPipelineConfigAdapter.from_file and to_yaml."""
        input_path = str(tmp_path / "input.json")
        output_path = str(tmp_path / "output.yaml")

        cli = self._make_cli_stub_with_config("lerobot", input_path, output_path)

        mock_adapter = MagicMock()
        mock_adapter.to_yaml.return_value = Path(output_path).resolve()

        with patch(
            "physicalai.config.lerobot.TrainPipelineConfigAdapter.from_file",
            return_value=mock_adapter,
        ) as mock_from_file:
            cli._run_config()

        mock_from_file.assert_called_once_with(input_path)
        mock_adapter.to_yaml.assert_called_once_with(output_path)

    def test_run_config_default_output_name(self, tmp_path: Path) -> None:
        """When output is None, generates default name '<stem>_physicalai.yaml'."""
        input_file = tmp_path / "my_config.json"
        input_file.write_text("{}")

        cli = self._make_cli_stub_with_config("lerobot", str(input_file), None)

        mock_adapter = MagicMock()
        mock_adapter.to_yaml.return_value = Path("my_config_physicalai.yaml").resolve()

        with patch(
            "physicalai.config.lerobot.TrainPipelineConfigAdapter.from_file",
            return_value=mock_adapter,
        ):
            cli._run_config()

        call_args = mock_adapter.to_yaml.call_args[0][0]
        assert call_args == "my_config_physicalai.yaml"

    def test_run_config_unsupported_format_raises(self) -> None:
        """Unsupported source_format → ValueError."""
        cli = self._make_cli_stub_with_config("pytorch", "input.json")

        with pytest.raises(ValueError, match="Unsupported source format"):
            cli._run_config()

    def test_run_config_no_subcommand_raises(self) -> None:
        """subcommand=None → ValueError."""
        from physicalai.cli.cli import CLI

        cli = object.__new__(CLI)
        cli._temp_config_file = None
        cli.subcommand = None

        with pytest.raises(ValueError, match="No subcommand specified"):
            cli._run_config()

    def test_run_config_missing_config_raises(self) -> None:
        """config.get returns None → ValueError."""
        from physicalai.cli.cli import CLI

        cli = object.__new__(CLI)
        cli._temp_config_file = None
        cli.subcommand = "config"
        cli.config = MagicMock()
        cli.config.get.return_value = None

        with pytest.raises(ValueError, match="config subcommand configuration not found"):
            cli._run_config()
