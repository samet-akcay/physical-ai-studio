# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Manifest schema and loader for exported model packages.

A manifest describes everything needed to reconstruct an inference
pipeline from a directory of exported artifacts: which runner to
use, which adapter, what shapes the hardware exposes, etc.

The on-disk format is ``manifest.json``.  The schema follows the
``class_path`` + ``init_args`` convention (jsonargparse-style) so that
domain layers can specify their own components without inferencekit
needing to know about them.

Backward compatibility
    If only a legacy ``metadata.yaml`` / ``metadata.json`` is present
    the loader transparently upgrades it to the manifest dataclass via
    :func:`Manifest.from_legacy_metadata`.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------- #
# Schema version
# ---------------------------------------------------------------------------- #
MANIFEST_VERSION = "1.0"
MANIFEST_FORMAT = "policy_package"


# ---------------------------------------------------------------------------- #
# Leaf dataclasses — one per manifest section
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Shape and dtype descriptor for a single tensor (state, action, image …).

    Attributes:
        shape: Tensor shape (no batch dimension).
        dtype: Numpy-compatible dtype string, e.g. ``"float32"``, ``"uint8"``.
    """

    shape: list[int]
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        """Create a TensorSpec from a plain dict.

        Returns:
            Parsed TensorSpec.
        """
        return cls(shape=list(data["shape"]), dtype=data.get("dtype", "float32"))


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """Robot hardware descriptor — state and action spaces.

    Attributes:
        name: Logical name, e.g. ``"main"``.
        type: Robot model string (informational), e.g. ``"Koch v1.1"``.
        state: Expected state tensor spec.
        action: Expected action tensor spec.
    """

    name: str
    type: str = ""
    state: TensorSpec | None = None
    action: TensorSpec | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotSpec:
        """Create a RobotSpec from a plain dict.

        Returns:
            Parsed RobotSpec.
        """
        return cls(
            name=data["name"],
            type=data.get("type", ""),
            state=TensorSpec.from_dict(data["state"]) if "state" in data else None,
            action=TensorSpec.from_dict(data["action"]) if "action" in data else None,
        )


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """Camera descriptor — image shape and dtype.

    Attributes:
        name: Logical name, e.g. ``"top"``, ``"wrist"``.
        shape: ``(C, H, W)`` tensor shape.
        dtype: Numpy-compatible dtype string.
    """

    name: str
    shape: list[int] = field(default_factory=list)
    dtype: str = "uint8"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraSpec:
        """Create a CameraSpec from a plain dict.

        Returns:
            Parsed CameraSpec.
        """
        return cls(
            name=data["name"],
            shape=list(data.get("shape", [])),
            dtype=data.get("dtype", "uint8"),
        )


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """Policy identity section.

    Attributes:
        name: Human-readable policy name, e.g. ``"act"``.
        kind: Runner kind hint.  Built-in models use ``"single_pass"``
            or ``"action_chunking"``.  Custom models specify a ``runner``
            section instead.
        class_path: Fully-qualified training class, e.g.
            ``"physicalai.policies.act.ACT"``.  Informational — the
            inference runtime does not import this.
    """

    name: str = ""
    kind: str = "single_pass"
    class_path: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicySpec:
        """Create a PolicySpec from a plain dict.

        Returns:
            Parsed PolicySpec.
        """
        return cls(
            name=data.get("name", ""),
            kind=data.get("kind", "single_pass"),
            class_path=data.get("class_path", ""),
        )


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A ``class_path`` + ``init_args`` pair for dynamic instantiation.

    Used for runners, adapters, preprocessors, and postprocessors.

    Attributes:
        class_path: Fully-qualified class path, e.g.
            ``"physicalai.inference.runners.SinglePass"``.
        init_args: Keyword arguments forwarded to the class constructor.
    """

    class_path: str
    init_args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentSpec:
        """Create a ComponentSpec from a plain dict.

        Returns:
            Parsed ComponentSpec.
        """
        return cls(
            class_path=data["class_path"],
            init_args=dict(data.get("init_args", {})),
        )

    def instantiate(self) -> Any:  # noqa: ANN401
        """Import the class and instantiate with ``init_args``.

        Returns:
            An instance of the class specified by ``class_path``.
        """
        module_path, class_name = self.class_path.rsplit(".", maxsplit=1)
        module = importlib.import_module(module_path)
        cls_obj = getattr(module, class_name)

        # Handle nested ComponentSpec in init_args (e.g. ActionChunking wrapping SinglePass)
        resolved_args: dict[str, Any] = {}
        for key, value in self.init_args.items():
            if isinstance(value, dict) and "class_path" in value:
                resolved_args[key] = ComponentSpec.from_dict(value).instantiate()
            else:
                resolved_args[key] = value

        return cls_obj(**resolved_args)


# ---------------------------------------------------------------------------- #
# Top-level Manifest
# ---------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Manifest:
    """Parsed manifest for an exported model package.

    Attributes:
        format: Always ``"policy_package"`` for this schema.
        version: Schema version string.
        policy: Policy identity (name, kind, training class).
        artifacts: Mapping of backend name → filename,
            e.g. ``{"onnx": "model.onnx"}``.
        runner: Optional dynamic runner specification.
        adapter: Optional dynamic adapter specification.
        robots: Hardware descriptors for robot state/action.
        cameras: Hardware descriptors for camera inputs.
        extra: Catch-all for domain-specific keys not covered above.
    """

    format: str = MANIFEST_FORMAT
    version: str = MANIFEST_VERSION
    policy: PolicySpec = field(default_factory=PolicySpec)
    artifacts: dict[str, str] = field(default_factory=dict)
    runner: ComponentSpec | None = None
    adapter: ComponentSpec | None = None
    robots: list[RobotSpec] = field(default_factory=list)
    cameras: list[CameraSpec] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Build a ``Manifest`` from a raw JSON dict.

        Unknown top-level keys are collected into ``extra`` so that
        domain layers can store additional data without schema changes.

        Returns:
            Parsed Manifest.
        """
        known_keys = {
            "format",
            "version",
            "policy",
            "artifacts",
            "runner",
            "adapter",
            "robots",
            "cameras",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            format=data.get("format", MANIFEST_FORMAT),
            version=data.get("version", MANIFEST_VERSION),
            policy=PolicySpec.from_dict(data["policy"]) if "policy" in data else PolicySpec(),
            artifacts=dict(data.get("artifacts", {})),
            runner=ComponentSpec.from_dict(data["runner"]) if "runner" in data else None,
            adapter=ComponentSpec.from_dict(data["adapter"]) if "adapter" in data else None,
            robots=[RobotSpec.from_dict(r) for r in data.get("robots", [])],
            cameras=[CameraSpec.from_dict(c) for c in data.get("cameras", [])],
            extra=extra,
        )

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load a manifest from a JSON file or directory.

        If *path* is a directory, looks for ``manifest.json`` inside it.

        Args:
            path: Path to ``manifest.json`` or a directory containing it.

        Returns:
            Parsed ``Manifest`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        p = Path(path)
        if p.is_dir():
            p /= "manifest.json"
        if not p.exists():
            msg = f"Manifest not found: {p}"
            raise FileNotFoundError(msg)
        with p.open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_legacy_metadata(cls, metadata: dict[str, Any]) -> Manifest:
        """Upgrade a legacy ``metadata.yaml`` dict to a ``Manifest``.

        Maps flat keys (``policy_class``, ``backend``, ``use_action_queue``,
        ``chunk_size``) into the structured manifest schema so that
        downstream code can use a single Manifest type regardless of the
        on-disk format.

        Args:
            metadata: Dict loaded from ``metadata.yaml`` or ``metadata.json``.

        Returns:
            A ``Manifest`` populated with as much information as the legacy
            format provides.
        """
        policy_class = metadata.get("policy_class", "")
        policy_name = _policy_name_from_class_path(policy_class)

        use_action_queue = metadata.get("use_action_queue", False)
        chunk_size = metadata.get("chunk_size", 1)

        kind = "action_chunking" if use_action_queue else "single_pass"

        runner = _build_runner_spec(kind, chunk_size)

        backend = metadata.get("backend", "")
        artifacts: dict[str, str] = {}
        if backend:
            artifacts[backend] = ""  # filename unknown from legacy metadata

        # Preserve everything in extra for backward compat
        extra = dict(metadata)

        return cls(
            policy=PolicySpec(name=policy_name, kind=kind, class_path=policy_class),
            artifacts=artifacts,
            runner=runner,
            robots=[],
            cameras=[],
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest to a plain dict suitable for ``json.dump``.

        Returns:
            JSON-serialisable dict.
        """
        data: dict[str, Any] = {
            "format": self.format,
            "version": self.version,
            "policy": {
                "name": self.policy.name,
                "kind": self.policy.kind,
                "class_path": self.policy.class_path,
            },
            "artifacts": self.artifacts,
        }

        if self.runner is not None:
            data["runner"] = {
                "class_path": self.runner.class_path,
                "init_args": self.runner.init_args,
            }

        if self.adapter is not None:
            data["adapter"] = {
                "class_path": self.adapter.class_path,
                "init_args": self.adapter.init_args,
            }

        if self.robots:
            data["robots"] = [_robot_to_dict(r) for r in self.robots]

        if self.cameras:
            data["cameras"] = [_camera_to_dict(c) for c in self.cameras]

        return data

    def save(self, path: str | Path) -> None:
        """Write the manifest as ``manifest.json``.

        Args:
            path: File path (typically ``export_dir / "manifest.json"``).
        """
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

RUNNER_CLASS_PATHS: dict[str, str] = {
    "single_pass": "physicalai.inference.runners.SinglePass",
    "action_chunking": "physicalai.inference.runners.ActionChunking",
}


def _build_runner_spec(kind: str, chunk_size: int = 1) -> ComponentSpec:
    """Build a ``ComponentSpec`` for a built-in runner kind.

    Args:
        kind: One of ``"single_pass"`` or ``"action_chunking"``.
        chunk_size: Chunk size for action-chunking runners.

    Returns:
        A ``ComponentSpec`` that can be instantiated later.
    """
    class_path = RUNNER_CLASS_PATHS.get(kind, RUNNER_CLASS_PATHS["single_pass"])

    if kind == "action_chunking":
        return ComponentSpec(
            class_path=class_path,
            init_args={
                "runner": {
                    "class_path": RUNNER_CLASS_PATHS["single_pass"],
                    "init_args": {},
                },
                "chunk_size": chunk_size,
            },
        )
    return ComponentSpec(class_path=class_path, init_args={})


def _policy_name_from_class_path(class_path: str) -> str:
    """Extract a short policy name from a fully-qualified class path.

    ``"physicalai.policies.act.policy.ACT"`` → ``"act"``

    Args:
        class_path: Dotted class path string.

    Returns:
        Short lowercase name, or empty string if extraction fails.
    """
    parts = class_path.lower().split(".")
    min_parts = 3
    if len(parts) >= min_parts:
        return parts[-2]
    return ""


def _robot_to_dict(robot: RobotSpec) -> dict[str, Any]:
    """Serialize a RobotSpec to a plain dict.

    Returns:
        JSON-serialisable dict.
    """
    data: dict[str, Any] = {"name": robot.name}
    if robot.type:
        data["type"] = robot.type
    if robot.state is not None:
        data["state"] = {"shape": robot.state.shape, "dtype": robot.state.dtype}
    if robot.action is not None:
        data["action"] = {"shape": robot.action.shape, "dtype": robot.action.dtype}
    return data


def _camera_to_dict(camera: CameraSpec) -> dict[str, Any]:
    """Serialize a CameraSpec to a plain dict.

    Returns:
        JSON-serialisable dict.
    """
    data: dict[str, Any] = {"name": camera.name}
    if camera.shape:
        data["shape"] = camera.shape
    if camera.dtype != "uint8":
        data["dtype"] = camera.dtype
    return data
