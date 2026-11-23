# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin Engine helpers layered directly on ``physicalai.isolate``."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # GPU probes use argv lists with shell disabled.  # nosec B404
import tempfile
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from physicalai import isolate

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ._types import WorkerEnv

logger = logging.getLogger(__name__)
_MARKER = ".physicalai-engine.json"
_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_.-]{1,128}")
# One CUDA profile; each environment maps it to its own extra (for example ``cu128`` or ``cu132``).
_ACCELERATOR_ALIASES = {"cpu": "cpu", "cuda": "cuda", "cu128": "cuda", "cu132": "cuda", "xpu": "xpu"}
_PCI_DISPLAY_CLASS = 0x03
_PCI_VENDOR_INTEL = "0x8086"
_PCI_VENDOR_NVIDIA = "0x10de"


@dataclass(frozen=True)
class CachedEnvironment:
    """Display metadata for one Engine-managed ``physicalai.isolate`` cache entry."""

    framework: str
    name: str
    accelerator: str
    key: str
    path: Path
    size_bytes: int = 0


def resolve_accelerator(accelerator: str, *, allowed: tuple[str, ...] | None = None) -> str:
    """Resolve ``auto`` or a device name to one supported worker profile.

    Automatic selection checks both the host torch runtime and physical GPU
    hardware, so a CPU-only host build does not hide an available GPU.

    Returns:
        Canonical accelerator profile: ``cpu``, ``cuda``, or ``xpu``.

    Raises:
        ValueError: If the requested profile is unknown, unsupported, or conflicts with detected GPU hardware.
    """
    normalized_allowed = None if allowed is None else {_canonical_accelerator(value) for value in allowed}
    if accelerator.strip().lower() == "auto":
        return _automatic_accelerator(normalized_allowed)

    selected = _canonical_accelerator(accelerator)
    if normalized_allowed is not None and selected not in normalized_allowed:
        msg = f"Accelerator {accelerator!r} is not supported by this engine environment."
        raise ValueError(msg)
    return selected


def _automatic_accelerator(allowed: set[str] | None) -> str:
    if allowed is not None and len(allowed) == 1:
        return next(iter(allowed))
    runtime = _torch_accelerators()
    hardware = _hardware_accelerators()
    order = ("xpu", "cuda")
    candidates = [device for device in order if runtime.get(device, False)]
    if not candidates:
        candidates = [device for device in order if device in hardware]
    if allowed is None:
        return candidates[0] if candidates else "cpu"
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    if candidates:
        detected = ", ".join("XPU" if candidate == "xpu" else "CUDA" for candidate in candidates)
        supported = ", ".join(sorted(allowed))
        msg = f"Detected {detected} hardware, but this engine environment supports only {supported}."
        raise ValueError(msg)
    for fallback in ("cpu", "xpu", "cuda"):
        if fallback in allowed:
            return fallback
    msg = "The worker environment does not declare a supported accelerator extra."
    raise ValueError(msg)


def _canonical_accelerator(accelerator: str) -> str:
    try:
        return _ACCELERATOR_ALIASES[accelerator.lower()]
    except KeyError:
        supported = ", ".join((*_ACCELERATOR_ALIASES, "auto"))
        msg = f"Unsupported accelerator {accelerator!r}; choose one of {supported}."
        raise ValueError(msg) from None


def _torch_accelerators() -> dict[str, bool]:
    try:
        import torch  # ruff: ignore[import-outside-top-level]  # Torch is optional when listing engines.
    except (ImportError, OSError, RuntimeError):
        return {"xpu": False, "cuda": False}

    def available(name: str) -> bool:
        try:
            backend = getattr(torch, name, None)
            is_available = getattr(backend, "is_available", None)
            return bool(backend is not None and callable(is_available) and is_available())
        except (AttributeError, OSError, RuntimeError, TypeError):
            return False

    return {"xpu": available("xpu"), "cuda": available("cuda")}


def _hardware_accelerators(devices: Path = Path("/sys/bus/pci/devices")) -> set[str]:
    detected = _pci_gpu_hardware(devices)
    if _command_reports_gpu("xpu-smi", ("discovery",), r"\bGPU\s*#?\s*\d+\b"):
        detected.add("xpu")
    if _command_reports_gpu("nvidia-smi", ("-L",), r"\bGPU\s+\d+\s*:"):
        detected.add("cuda")
    return detected


def _pci_gpu_hardware(devices: Path) -> set[str]:
    detected: set[str] = set()
    if not devices.is_dir():
        return detected
    try:
        entries = tuple(devices.iterdir())
    except OSError:
        return detected
    for device in entries:
        try:
            vendor = (device / "vendor").read_text(encoding="ascii").strip().lower()
            class_code = int((device / "class").read_text(encoding="ascii").strip(), 16)
        except (OSError, ValueError):
            continue
        if class_code >> 16 != _PCI_DISPLAY_CLASS:
            continue
        if vendor == _PCI_VENDOR_INTEL:
            detected.add("xpu")
        elif vendor == _PCI_VENDOR_NVIDIA:
            detected.add("cuda")
    return detected


def _command_reports_gpu(command: str, arguments: tuple[str, ...], pattern: str) -> bool:
    executable = shutil.which(command)
    if executable is None:
        return False
    try:
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # nosec B603
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and re.search(pattern, result.stdout + result.stderr, re.IGNORECASE) is not None


def locked_version(env: WorkerEnv, project: Path | None = None) -> Version:
    """Read the exact worker package version from ``uv.lock``.

    Returns:
        The exact locked package version.

    Raises:
        ValueError: If the lock is unavailable, invalid TOML, or does not pin one matching package.
    """
    lock_bytes = _lock_bytes(env, project)
    try:
        lock = tomllib.loads(lock_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        msg = f"Worker lock for {env.framework!r} is not valid TOML."
        raise ValueError(msg) from exc

    package_name = _normalize_distribution_name(env.package)
    versions = {
        Version(package["version"])
        for package in lock.get("package", [])
        if isinstance(package, dict)
        and isinstance(package.get("name"), str)
        and _normalize_distribution_name(package["name"]) == package_name
        and isinstance(package.get("version"), str)
    }
    if len(versions) != 1:
        msg = (
            f"Expected one locked version of {env.package!r} for {env.framework!r}, found {sorted(map(str, versions))}."
        )
        raise ValueError(msg)
    return next(iter(versions))


def supported_versions(env: WorkerEnv) -> SpecifierSet | None:
    """Read the package range declared by the plugin, if it declares one.

    Returns:
        The declared specifier, or ``None`` when the plugin declares no requirement.

    Raises:
        LookupError: If the plugin declares more than one unmarked requirement on the package.
    """
    package_name = _normalize_distribution_name(env.package)
    requirements = []
    for raw_requirement in metadata.requires(env.distribution) or []:
        requirement = Requirement(raw_requirement)
        if _normalize_distribution_name(requirement.name) != package_name:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        requirements.append(requirement.specifier)

    if not requirements:
        return None
    if len(requirements) != 1:
        msg = (
            f"Expected one unmarked {env.package!r} requirement in distribution "
            f"{env.distribution!r}, found {len(requirements)}."
        )
        raise LookupError(msg)
    return SpecifierSet(str(requirements[0]))


def prepare(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> isolate.PreparedEnvironment:
    """Prepare or reuse the isolated worker environment via ``physicalai.isolate``.

    Returns:
        The Runtime :class:`physicalai.isolate.PreparedEnvironment`.
    """
    selected = resolve_accelerator(accelerator, allowed=env.accelerators)
    for value, label in ((env.framework, "framework"), (env.name, "name"), (env.distribution, "distribution")):
        _validate_component(value, label)
    prepared = isolate.prepare(
        env.to_isolate(selected),
        provider=isolate.UvProvider(cache_dir=_cache_root(), uv=_installed_uv()),
        offline=offline,
    )
    _write_marker(
        prepared.path,
        {
            "framework": env.framework,
            "name": env.name,
            "accelerator": selected,
            "key": prepared.key,
        },
    )
    return prepared


def list_prepared() -> list[CachedEnvironment]:
    """List Engine-managed ``physicalai.isolate`` cache entries and their on-disk size.

    Returns:
        Cached environments sorted by framework, name, accelerator, and path.
    """
    prepared: list[CachedEnvironment] = []
    for root, marker in _markers():
        isolate_marker = root / ".physicalai-isolate.json"
        if isolate_marker.is_symlink() or not isolate_marker.is_file():
            continue
        if any((root / name).is_symlink() for name in ("project", "venv")):
            continue
        entry = isolate.PreparedEnvironment(path=root, key=marker["key"], profile=isolate.SyncProfile())
        if not entry.project.is_dir() or not entry.venv.is_dir() or not entry.python.is_file():
            continue
        prepared.append(
            CachedEnvironment(
                framework=marker["framework"],
                name=marker["name"],
                accelerator=marker["accelerator"],
                key=marker["key"],
                path=root,
                size_bytes=_directory_size(root),
            ),
        )
    return sorted(prepared, key=lambda item: (item.framework, item.name, item.accelerator, str(item.path)))


def prune_prepared(current: Iterable[WorkerEnv]) -> list[Path]:
    """Remove Engine-tagged cache entries no installed engine currently declares.

    Runtime-owned isolate cache entries without an Engine tag are left untouched.

    Returns:
        Paths of removed cache directories.
    """
    cache = _cache_root()
    if not cache.is_dir():
        return []

    current_keys = {env.to_isolate(acc).key() for env in current for acc in env.accelerators}
    removed: list[Path] = []
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        marker_path = root / _MARKER
        if marker_path.is_symlink() or not marker_path.is_file():
            continue
        try:
            marker = _read_marker(root)
            stale = marker["key"] not in current_keys
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stale = True
        if stale:
            shutil.rmtree(root)
            removed.append(root)
    return sorted(removed)


def prune() -> list[Path]:
    """Prune cached environments that installed engines no longer declare.

    Returns:
        Paths of removed cache directories.
    """
    from . import Engine  # ruff: ignore[import-outside-top-level]  # Avoid circular import at package init.

    current: list[WorkerEnv] = []
    for name in Engine.names():
        try:
            current.extend(Engine.get(name).environments())
        except (ImportError, TypeError, ValueError):
            continue
    return prune_prepared(current)


def _cache_root() -> Path:
    configured = os.environ.get("PHYSICALAI_ISOLATE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "physicalai" / "isolate").resolve()


def _installed_uv() -> str | None:
    """Return the uv executable installed alongside this Python environment, if present.

    Returns:
        Path to the bundled uv binary, or ``None`` to let isolate look up ``uv`` on ``PATH``.
    """
    try:
        from uv import find_uv_bin  # ruff: ignore[import-outside-top-level]  # Optional at import time.

        return find_uv_bin()
    except (ImportError, FileNotFoundError):
        return None


def _lock_bytes(env: WorkerEnv, project: Path | None) -> bytes:
    if project is not None:
        lock = project / "uv.lock"
        if not lock.is_file():
            msg = f"Worker project for {env.framework!r} does not include uv.lock."
            raise FileNotFoundError(msg)
        return lock.read_bytes()
    read_lock = getattr(env.source, "lock", None)
    lock_data = read_lock() if callable(read_lock) else None
    if lock_data is None:
        msg = f"The {env.framework!r} {env.name!r} environment's lock is available only after preparation."
        raise ValueError(msg)
    return lock_data


def _markers() -> Iterator[tuple[Path, dict[str, str]]]:
    cache = _cache_root()
    if not cache.is_dir():
        return
    for root in sorted(cache.iterdir()):
        if root.is_symlink() or not root.is_dir() or not (root / _MARKER).is_file():
            continue
        try:
            marker = _read_marker(root)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Skipping worker environment %s with an invalid Engine marker; `physicalai engine prune` removes it.",
                root,
            )
            continue
        yield root, marker


def _write_marker(root: Path, marker: dict[str, str]) -> None:
    marker_path = root / _MARKER
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as file:
        json.dump(marker, file, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(marker_path)


def _read_marker(root: Path) -> dict[str, str]:
    marker_path = root / _MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        msg = f"Worker environment marker not found: {marker_path}"
        raise FileNotFoundError(msg)
    raw = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Worker environment marker is not an object: {marker_path}"
        raise TypeError(msg)
    fields: dict[str, str] = {}
    for key in ("framework", "name", "accelerator", "key"):
        value = raw.get(key)
        if not isinstance(value, str):
            msg = f"Worker environment marker has an invalid {key!r} field."
            raise TypeError(msg)
        fields[key] = value
    return fields


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if not path.is_symlink() and path.is_file())


def _validate_component(value: str, label: str) -> None:
    if _COMPONENT_RE.fullmatch(value) is None:
        msg = f"Invalid worker environment {label}: {value!r}."
        raise ValueError(msg)


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


__all__ = [
    "CachedEnvironment",
    "list_prepared",
    "locked_version",
    "prepare",
    "prune",
    "prune_prepared",
    "resolve_accelerator",
    "supported_versions",
]
