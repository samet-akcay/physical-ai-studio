# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Framework-neutral APIs for framework and model-repository engines.

Engine classes are discovered lazily from the ``physicalai.engines`` entry
point group. Every engine packages models for Runtime; :class:`TrainingEngine`
subclasses also train. Worker environments delegate isolation, caching, and
process launch to ``physicalai.isolate``. Importing this module does not import
Lightning, LeRobot, PyTorch, or Runtime inference adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, overload

from . import _registry
from ._environment import (
    CachedEnvironment,
    list_prepared,
    locked_version,
    prepare,
    prune,
    resolve_accelerator,
    supported_versions,
)
from ._runner import run_worker
from ._types import Capabilities, Run, TrainHooks, TrainingCancelledError, WorkerEnv

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from typing import Any

    from physicalai.inference.model import InferenceModel
    from physicalai_lerobot_policy_plugin.engine import LeRobotEngine

    from physicalai.train.trainer import Trainer


class Engine(ABC):
    """Common contract for one framework or model repository.

    Every engine packages models for Runtime and reports what it provides.
    Engines that also train subclass :class:`TrainingEngine`.
    """

    framework: ClassVar[str]
    env: ClassVar[WorkerEnv | None] = None
    stock_export_formats: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def export(
        cls,
        source: Run | str | Path,
        output_path: str | Path,
        backend: str | None = None,
    ) -> Path:
        """Package a native checkpoint for Runtime inference."""

    @classmethod
    @abstractmethod
    def policies(cls) -> list[str]:
        """List policy identifiers this engine provides."""

    @classmethod
    def environments(cls) -> tuple[WorkerEnv, ...]:
        """Return the worker environments declared by this engine.

        Returns:
            A one-element tuple containing ``env`` when managed, else an empty tuple.
        """
        return (cls.env,) if cls.env is not None else ()

    @classmethod
    def environment(cls) -> WorkerEnv:
        """Return the engine's managed worker environment.

        Returns:
            The declared worker environment.

        Raises:
            ValueError: If the engine does not use a managed worker environment.
        """
        if cls.env is None:
            msg = f"Engine {cls.framework!r} does not use a managed worker environment."
            raise ValueError(msg)
        return cls.env

    @classmethod
    def capabilities(cls, policy: str) -> Capabilities:
        """Return training accelerators and export formats supported for *policy*.

        Training accelerators are reported only by :class:`TrainingEngine` subclasses.

        Raises:
            ValueError: If *policy* is not provided by this engine.
        """
        if policy not in cls.policies():
            msg = f"Unknown {cls.framework} policy {policy!r}."
            raise ValueError(msg)

        accelerators: set[str] = set()
        if issubclass(cls, TrainingEngine):
            accelerators.update(getattr(cls, "training_accelerators", ()))
            if cls.env is not None:
                accelerators.update(cls.env.accelerators)
        return Capabilities(
            training_accelerators=tuple(sorted(accelerators)),
            export_formats=tuple(sorted(cls.stock_export_formats)),
        )

    @overload
    @classmethod
    def create(
        cls,
        name: Literal["physicalai"],
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> Trainer: ...

    @overload
    @classmethod
    def create(
        cls,
        name: Literal["lerobot"],
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> LeRobotEngine: ...

    @overload
    @classmethod
    def create(
        cls,
        name: str,
        /,
        **options: Any,  # ruff: ignore[any-type]  # Engine constructor options are framework-specific.
    ) -> Engine: ...

    @classmethod
    def create(cls, name: str, /, **options: Any) -> Engine:
        """Instantiate the engine registered as *name*.

        Returns:
            An instance of the selected engine.
        """
        engine_class = cls.get(name)
        return engine_class(**options)

    @overload
    @classmethod
    def get(cls, name: Literal["physicalai"], /) -> type[Trainer]: ...

    @overload
    @classmethod
    def get(cls, name: Literal["lerobot"], /) -> type[LeRobotEngine]: ...

    @overload
    @classmethod
    def get(cls, name: str, /) -> type[Engine]: ...

    @classmethod
    def get(cls, name: str, /) -> type[Engine]:
        """Return the engine class registered as *name*.

        Returns:
            The class registered under *name*.
        """
        return _registry.get_engine(name, Engine)

    @classmethod
    def names(cls) -> list[str]:
        """List installed engine identifiers without importing engine modules.

        Returns:
            Installed engine identifiers in sorted order.
        """
        return _registry.names()

    @staticmethod
    def load(package: str | Path, **kwargs: Any) -> InferenceModel:  # ruff: ignore[any-type]  # Runtime accepts adapter options.
        """Load a Runtime policy package through the Runtime API.

        Returns:
            The initialized Runtime inference model.
        """
        from physicalai.inference.model import InferenceModel as RuntimeInferenceModel  # ruff: ignore[import-outside-top-level]

        return RuntimeInferenceModel(package, **kwargs)


class TrainingEngine(Engine):
    """An engine that also trains through its framework's native recipe."""

    training_accelerators: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def cli(
        cls,
        argv: Sequence[str],
        *,
        interpreter: str | Path | None = None,
    ) -> int:
        """Run the framework's native training CLI and return its exit code."""

    @classmethod
    @abstractmethod
    def run(
        cls,
        settings: Mapping[str, Any],
        *,
        dataset: Path,
        output_dir: Path,
        device: str,
        hooks: TrainHooks | None = None,
        interpreter: str | Path | None = None,
    ) -> Run:
        """Run a headless training job from native settings and standard inputs."""

    @classmethod
    @abstractmethod
    def config_schema(cls, policy: str) -> dict[str, Any]:
        """Return the native training config schema for *policy*."""


__all__ = [
    "CachedEnvironment",
    "Capabilities",
    "Engine",
    "Run",
    "TrainHooks",
    "TrainingCancelledError",
    "TrainingEngine",
    "WorkerEnv",
    "list_prepared",
    "locked_version",
    "prepare",
    "prune",
    "resolve_accelerator",
    "run_worker",
    "supported_versions",
]
