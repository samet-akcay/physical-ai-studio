# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pinned GR00T sources and the worker environments built from them.

This module only declares pins. Core ``physicalai.engine`` materializes them:
it checks out the upstream commit, verifies and applies the Robotics AI Suite
patch series, checks the patched tree, and synchronizes the checkout's own
``uv.lock`` into a cached environment keyed by every pin.

The OpenVINO patches are published by the Robotics AI Suite, so they are
declared as :class:`~physicalai.engine.PublishedPatches`. If their maintainers
move them into this plugin, the declaration becomes
``BundledPatches(_PACKAGE / "patches" / "openvino", tree=...)``; nothing else changes.

Two environments come from the same upstream commit:

- ``default``: upstream Isaac-GR00T (Python 3.10, CUDA 12.8) for training and
  native PyTorch inference;
- ``robotics-ai-suite-openvino``: the same commit plus the Suite's OpenVINO
  patches (Python 3.12, CPU PyTorch, OpenVINO) for export and inference.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from physicalai.engine import Enablement, GitProject, PatchFile, PublishedPatches, WorkerEnv, prepare_environment

from .package import BACKEND as FRAMEWORK
from .package import FORMAT_NATIVE, FORMAT_OPENVINO, POLICY

DISTRIBUTION = "physicalai-gr00t-policy-plugin"
PROJECT_ENV = "PHYSICALAI_WORKER_PROJECT"

_PACKAGE = resources.files("physicalai_gr00t_policy_plugin")
TRAIN_WORKER = _PACKAGE / "_worker" / "gr00t_train_worker.py"
EXPORT_WORKER = _PACKAGE / "_worker" / "gr00t_export_worker.py"
INFER_WORKER = _PACKAGE / "_worker" / "gr00t_infer_worker.py"

UPSTREAM = GitProject(
    repository="https://github.com/NVIDIA/Isaac-GR00T.git",
    revision="3df8b3825d67f755e69141446f4315f281b9b7e6",
)

ROBOTICS_AI_SUITE_OPENVINO = PublishedPatches(
    repository="https://github.com/open-edge-platform/robotics-ai-suite.git",
    revision="2960959ae6cc6d45150316ed6612f686c2d915b7",
    directory="src/pipelines/gr00t-n1d7-ov/patches",
    files=(
        PatchFile(
            "0001-Add-OpenVINO-model-convert-inference-code.patch",
            "2e7d6355b948926a965ee30df1bc2d7623a51233547d90e8eea16a967188ed2d",
        ),
        PatchFile(
            "0002-Add-lazy-set-static-shape.patch",
            "da7c5266e69362ec4f81bfcf995babfbbeb7f8ff3005fc002d7974137dc86fa0",
        ),
        PatchFile(
            "0003-Add-static-LLM-seq_len-padding-via-llm-lang-tokens.patch",
            "2d47606257fff4214825a8daf82056170c6111dc80f51f674b6c1310eba880e6",
        ),
        PatchFile(
            "0004-Enable-gr00t-n1.7-sonic-pipeline-on-ov.patch",
            "8ea2712b3c3235582a1c491fe81dfda4bd8c4a67be0c37b405f75049e077e912",
        ),
        PatchFile(
            "0005-Refresh-OpenVINO-export-and-inference-READMEs.patch",
            "e91d474388ae3a8ceb815c1f777952dceecb45e862edd77ab578ed08714d9a3d",
        ),
        PatchFile(
            "0006-Enable-3-camera-in-embodiment-tag-UNITREE_G1_SONIC.patch",
            "703fccdef8504b1fe3885b90051fa022e208e2decf97a3a210135d5e37d3e6a3",
        ),
        PatchFile(
            "0007-Address-Bandit-findings.patch",
            "b867bb2e27bfc569457f192b73c79f613f79e4fdf20a3da6bbece41524488227",
        ),
        PatchFile(
            "0008-single-OV-server-path-honor-llm_static_seq_len-paddi.patch",
            "ceb5527f9f558260dccf8617339d24b0e45d4543b228da183e1ec93de489cb09",
        ),
    ),
    tree="d657bcb89b8937495b8c849d7cfb8c9f3d1e5601",
    license="Apache-2.0",
    mirror=_PACKAGE / "patches" / "robotics-ai-suite-gr00t-n1d7-ov",
)

NATIVE = WorkerEnv(
    framework=FRAMEWORK,
    project=UPSTREAM,
    worker=TRAIN_WORKER,
    distribution=DISTRIBUTION,
    package="gr00t",
    accelerators=("cu128",),
    python="3.10",
    extras={"cu128": ()},
)

OPENVINO = WorkerEnv(
    framework=FRAMEWORK,
    name="robotics-ai-suite-openvino",
    project=GitProject(UPSTREAM.repository, UPSTREAM.revision, patches=ROBOTICS_AI_SUITE_OPENVINO),
    worker=EXPORT_WORKER,
    distribution=DISTRIBUTION,
    package="gr00t",
    accelerators=("cpu",),
    python="3.12",
    extras={"cpu": ()},
)

OPENVINO_ENABLEMENT = Enablement(
    name=OPENVINO.name,
    provides={POLICY: {"export": (FORMAT_OPENVINO,)}},
    upstream=(
        f"{UPSTREAM.repository}@{UPSTREAM.revision} + "
        f"{ROBOTICS_AI_SUITE_OPENVINO.repository}@{ROBOTICS_AI_SUITE_OPENVINO.revision}:"
        f"{ROBOTICS_AI_SUITE_OPENVINO.directory}"
    ),
    env=OPENVINO,
)

ENVIRONMENTS: dict[str, WorkerEnv] = {NATIVE.name: NATIVE, OPENVINO.name: OPENVINO}
PACKAGE_ENVIRONMENTS: dict[str, WorkerEnv] = {FORMAT_NATIVE: NATIVE, FORMAT_OPENVINO: OPENVINO}


def resolve(
    env: WorkerEnv,
    *,
    interpreter: str | Path | None = None,
    project: str | Path | None = None,
) -> tuple[Path, Path, bool]:
    """Return the interpreter and checkout for *env*, preparing it when needed.

    Args:
        env: Environment to use.
        interpreter: User-managed interpreter that bypasses provisioning.
        project: Checkout used by *interpreter*; defaults to ``$PHYSICALAI_WORKER_PROJECT``.

    Returns:
        Interpreter, upstream checkout, and whether core manages the environment.

    Raises:
        ValueError: If a user-managed interpreter has no checkout.
    """
    if interpreter is None:
        prepared = prepare_environment(env)
        return prepared.python, prepared.project, True
    checkout = project if project is not None else os.environ.get(PROJECT_ENV)
    if not checkout:
        msg = f"Pass project= or set {PROJECT_ENV} to the Isaac-GR00T checkout used by {interpreter}."
        raise ValueError(msg)
    return Path(interpreter).expanduser(), Path(checkout).expanduser(), False


def provenance(env: WorkerEnv) -> dict[str, object]:
    """Describe an environment for package and run records.

    Returns:
        Environment name, content key, and where the source and patches come from.
    """
    return {"framework": env.framework, "name": env.name, "key": env.key(), "source": env.source.provenance()}
