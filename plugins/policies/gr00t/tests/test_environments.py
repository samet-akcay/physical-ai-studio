# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the GR00T pins and the environments declared from them."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from physicalai.engine import GitProject

from physicalai_gr00t_policy_plugin import environments
from physicalai_gr00t_policy_plugin.engine import Gr00tEngine

_SHA1 = re.compile(r"[0-9a-f]{40}")


def test_patch_series_pins_the_robotics_ai_suite_revision_digests_and_tree() -> None:
    series = environments.ROBOTICS_AI_SUITE_OPENVINO
    assert series.revision == "2960959ae6cc6d45150316ed6612f686c2d915b7"
    assert series.tree == "d657bcb89b8937495b8c849d7cfb8c9f3d1e5601"
    assert [patch.name for patch in series.files] == sorted(patch.name for patch in series.files)
    assert len(series.files) == 8
    assert series.url(series.files[0]) == (
        "https://raw.githubusercontent.com/open-edge-platform/robotics-ai-suite/"
        "2960959ae6cc6d45150316ed6612f686c2d915b7/src/pipelines/gr00t-n1d7-ov/patches/"
        "0001-Add-OpenVINO-model-convert-inference-code.patch"
    )


def test_native_and_openvino_environments_share_the_upstream_commit() -> None:
    native, openvino = environments.NATIVE, environments.OPENVINO
    assert isinstance(native.source, GitProject)
    assert isinstance(openvino.source, GitProject)
    assert native.source.revision == openvino.source.revision == "3df8b3825d67f755e69141446f4315f281b9b7e6"
    assert native.source.layers == ()
    assert openvino.source.layers == (environments.ROBOTICS_AI_SUITE_OPENVINO,)
    assert (native.name, native.python, native.accelerators, native.extras_for("cu128")) == (
        "default",
        "3.10",
        ("cu128",),
        (),
    )
    assert (openvino.name, openvino.python, openvino.accelerators, openvino.extras_for("cpu")) == (
        "robotics-ai-suite-openvino",
        "3.12",
        ("cpu",),
        (),
    )
    assert native.key() != openvino.key()
    assert _SHA1.fullmatch(environments.UPSTREAM.revision)


def test_engine_declares_both_environments_through_core() -> None:
    assert Gr00tEngine.environments() == (environments.NATIVE, environments.OPENVINO)
    assert Gr00tEngine.environment("robotics-ai-suite-openvino") is environments.OPENVINO
    assert environments.OPENVINO_ENABLEMENT.env is environments.OPENVINO
    assert environments.PACKAGE_ENVIRONMENTS == {"native": environments.NATIVE, "openvino": environments.OPENVINO}


def test_provenance_records_environment_name_key_and_pins() -> None:
    record = environments.provenance(environments.OPENVINO)
    assert record["name"] == "robotics-ai-suite-openvino"
    assert record["key"] == environments.OPENVINO.key()
    assert record["source"]["patches"][0]["kind"] == "published"
    assert record["source"]["patches"][0]["tree"] == "d657bcb89b8937495b8c849d7cfb8c9f3d1e5601"


def test_resolve_prepares_managed_environments_and_accepts_user_checkouts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = type("Prepared", (), {"python": tmp_path / "python", "project": tmp_path / "project"})()
    monkeypatch.setattr(environments, "prepare_environment", lambda _env: prepared)
    assert environments.resolve(environments.NATIVE) == (tmp_path / "python", tmp_path / "project", True)

    monkeypatch.setenv("PHYSICALAI_WORKER_PROJECT", "/opt/gr00t")
    assert environments.resolve(environments.NATIVE, interpreter="/opt/py") == (
        Path("/opt/py"),
        Path("/opt/gr00t"),
        False,
    )
    monkeypatch.delenv("PHYSICALAI_WORKER_PROJECT")
    with pytest.raises(ValueError, match="Isaac-GR00T checkout"):
        environments.resolve(environments.NATIVE, interpreter="/opt/py")
