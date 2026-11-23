# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Lightweight shared types for framework engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from physicalai import isolate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from importlib.resources.abc import Traversable  # nosemgrep  # Python 3.10+ only.
    from pathlib import Path


class TrainingCancelledError(RuntimeError):
    """Raised when a training hook requests cancellation."""

    def __init__(self, step: int | None = None) -> None:
        self.step = step
        detail = f" after step {step}" if step is not None else ""
        super().__init__(f"Training cancelled{detail}.")


class TrainHooks(Protocol):
    """Progress and cancellation hooks shared by all native trainers."""

    def on_step(self, step: int, total: int | None, metrics: Mapping[str, float]) -> None:
        """Report one completed optimization step."""

    def should_stop(self) -> bool:
        """Return whether the current training run should stop."""


@dataclass(frozen=True)
class Run:
    """Result of a native framework training invocation."""

    framework: str
    output_dir: Path | None
    checkpoint: Path | None


@dataclass(frozen=True)
class Capabilities:
    """Validated training and export capabilities for one policy."""

    training_accelerators: tuple[str, ...] = ()
    export_formats: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerEnv:
    """Thin Engine descriptor over a ``physicalai.isolate.Environment``.

    ``project`` is a :class:`physicalai.isolate.ProjectSource`, or a directory
    containing ``pyproject.toml`` and ``uv.lock`` (wrapped as
    :class:`physicalai.isolate.BundledProject`). ``worker`` is the default
    standard-library-compatible script executed inside the isolated environment.

    Attributes:
        framework: Engine identifier that owns the environment.
        project: Source of the uv project.
        worker: Default worker script.
        distribution: Plugin distribution that declares the environment.
        package: Framework distribution installed in the environment; workers report its version.
        accelerators: Accelerator profiles the environment supports.
        name: Environment name, unique within its framework.
        python: Python version requested from uv, or ``None`` to follow the project.
        extras: uv extras per accelerator profile, for example ``{"cuda": ("cu132",)}``. Without it,
            a profile installs the extra named like the declared profile.
    """

    framework: str
    project: Path | Traversable | isolate.ProjectSource
    worker: Path | Traversable
    distribution: str
    package: str
    accelerators: tuple[str, ...] = ("cpu", "cuda", "xpu")
    name: str = "default"
    python: str | None = None
    extras: Mapping[str, tuple[str, ...]] | None = field(default=None, hash=False)

    @property
    def source(self) -> isolate.ProjectSource:
        """The Runtime isolation project source."""
        if isinstance(self.project, isolate.ProjectSource):
            return self.project
        return isolate.BundledProject(self.project)

    def extras_for(self, accelerator: str) -> tuple[str, ...]:
        """Return the uv extras that select *accelerator*.

        Profiles match by canonical name, so ``cuda`` and ``cu128`` name the same CUDA profile.

        Returns:
            Extra names passed to ``uv sync``.
        """
        from ._environment import _canonical_accelerator  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        wanted = _canonical_accelerator(accelerator)
        if self.extras is None:
            declared = (name for name in self.accelerators if _canonical_accelerator(name) == wanted)
            return (next(declared, accelerator),)
        for name, extras in self.extras.items():
            if _canonical_accelerator(name) == wanted:
                return tuple(extras)
        return ()

    def to_isolate(self, accelerator: str = "auto") -> isolate.Environment:
        """Build the ``physicalai.isolate.Environment`` for *accelerator*.

        Returns:
            Runtime isolation environment with the resolved Python and uv extras.
        """
        from ._environment import resolve_accelerator  # ruff: ignore[import-outside-top-level]  # Lazy to avoid an import cycle.

        selected = resolve_accelerator(accelerator, allowed=self.accelerators)
        profile = isolate.SyncProfile(python=self.python, extras=self.extras_for(selected))
        return isolate.Environment(source=self.source, profile=profile)


__all__ = [
    "Capabilities",
    "Run",
    "TrainHooks",
    "TrainingCancelledError",
    "WorkerEnv",
]
