# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Provision and inspect locked uv worker environments."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Invokes pinned uv with shell=False and a fixed subcommand.
import tempfile
import tomllib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from ._sources import ProjectSource
    from ._types import WorkerEnv

logger = logging.getLogger(__name__)
_MARKER = ".physicalai-engine.json"
_MARKER_VERSION = 2
_COMPONENT_RE = re.compile(r"[a-zA-Z0-9_.-]{1,128}")
_KEY_PREFIX = 12
_ACCELERATOR_ALIASES = {"cpu": "cpu", "cuda": "cu128", "cu128": "cu128", "xpu": "xpu"}
_PCI_DISPLAY_CLASS = 0x03
_PCI_VENDOR_INTEL = "0x8086"
_PCI_VENDOR_NVIDIA = "0x10de"


@dataclass(frozen=True)
class PreparedEnvironment:
    """A managed worker environment.

    Attributes:
        framework: Engine identifier.
        name: Environment name within the framework (``default`` or an enablement).
        accelerator: Accelerator profile that was synchronized.
        key: Content key of the environment's inputs.
        distribution: Plugin distribution that declared the environment.
        path: Environment root, containing ``project`` and ``venv``.
        size_bytes: On-disk size, when measured.
    """

    framework: str
    name: str
    accelerator: str
    key: str
    distribution: str
    path: Path
    size_bytes: int = 0

    @property
    def python(self) -> Path:
        """The environment's Python executable."""
        return _python_path(self.path / "venv")

    @property
    def project(self) -> Path:
        """The materialized uv project the environment was built from."""
        return self.path / "project"


def engine_cache_root() -> Path:
    """Return the user-owned cache root for managed worker environments."""
    configured = os.environ.get("PHYSICALAI_ENGINE_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return (cache_home / "physicalai" / "engines").resolve()


def resolve_accelerator(accelerator: str, *, allowed: tuple[str, ...] | None = None) -> str:
    """Resolve ``auto`` or a device name to one supported worker uv extra.

    Automatic selection checks both the host torch runtime and physical GPU
    hardware, so a CPU-only host build does not hide an available GPU.

    Returns:
        Canonical uv extra name: ``cpu``, ``cu128``, or ``xpu``.

    Raises:
        ValueError: If the requested extra is unknown, unsupported, or conflicts with detected GPU hardware.
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
    """Pick the best profile for the host among the environment's profiles.

    Returns:
        Canonical profile name.

    Raises:
        ValueError: If detected GPU hardware has no matching profile.
    """
    if allowed is not None and len(allowed) == 1:
        # A single-profile environment (for example, a CUDA-only upstream lock) has nothing to choose.
        return next(iter(allowed))
    runtime = _torch_accelerators()
    hardware = _hardware_accelerators()
    order = (("xpu", "xpu"), ("cuda", "cu128"))
    candidates = [extra for device, extra in order if runtime.get(device, False)]
    if not candidates:
        candidates = [extra for device, extra in order if device in hardware]
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
    for fallback in ("cpu", "xpu", "cu128"):
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


def _hardware_accelerators(pci_devices: Path = Path("/sys/bus/pci/devices")) -> set[str]:
    detected = _pci_gpu_hardware(pci_devices)
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
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
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


def uv_bin() -> str:
    """Return the executable bundled by the pinned ``uv`` Python package.

    Raises:
        RuntimeError: If the ``uv`` package is missing from the host environment.
    """
    try:
        from uv import find_uv_bin  # ruff: ignore[import-outside-top-level]  # Loaded only when an env is prepared.
    except ImportError as exc:
        msg = "Managed engine environments require the pinned 'uv' package; install physicalai-train first."
        raise RuntimeError(msg) from exc
    return str(find_uv_bin())


def lock_bytes(env: WorkerEnv, project: Path | None = None) -> bytes:
    """Read a worker lock from a prepared project or from a bundled source.

    Args:
        env: Worker environment.
        project: Materialized project directory. Required for sources whose
            lock exists only after materialization, such as Git projects.

    Returns:
        The raw ``uv.lock`` bytes.

    Raises:
        FileNotFoundError: If the project does not include a lock.
        ValueError: If the source has no lock before preparation and no project was given.
    """
    if project is not None:
        lock = project / "uv.lock"
        if not lock.is_file():
            msg = f"Worker project for {env.framework!r} does not include uv.lock."
            raise FileNotFoundError(msg)
        return lock.read_bytes()
    lock_data = env.source.lock()
    if lock_data is None:
        msg = f"The {env.framework!r} {env.name!r} environment's lock is available only after preparation."
        raise ValueError(msg)
    return lock_data


def lock_hash(env: WorkerEnv, project: Path | None = None) -> str:
    """Return the SHA-256 digest of a worker lock."""
    return hashlib.sha256(lock_bytes(env, project)).hexdigest()


def locked_version(env: WorkerEnv, project: Path | None = None) -> Version:
    """Read the exact worker package version from ``uv.lock``.

    Returns:
        The exact locked package version.

    Raises:
        ValueError: If the lock is invalid or does not pin exactly one matching package.
    """
    try:
        lock = tomllib.loads(lock_bytes(env, project).decode("utf-8"))
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

    Plugins that depend on a PyPI framework (LeRobot) declare its range. Plugins
    for frameworks pinned by Git commit declare none; their environments are
    identified by content key and exact lock instead.

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


def prepare(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> Path:
    """Create or synchronize the worker environment and return its interpreter.

    Returns:
        Path to the prepared worker Python executable.
    """
    return prepare_environment(env, accelerator, offline=offline).python


def prepare_environment(env: WorkerEnv, accelerator: str = "auto", *, offline: bool = False) -> PreparedEnvironment:
    """Create or synchronize the immutable worker environment for *env*.

    The source is materialized outside the caller's working tree, so a host
    project's ``[tool.uv]`` configuration cannot affect resolution, and
    ``--frozen`` makes the project's lock the only resolution source. The cache
    directory is keyed by framework, environment name, accelerator, and the
    content key of every input.

    Returns:
        The prepared environment.

    Raises:
        ValueError: If the worker framework, name, or accelerator is invalid.
    """
    selected = resolve_accelerator(accelerator, allowed=env.accelerators)
    for value, label in ((env.framework, "framework"), (env.name, "name"), (env.distribution, "distribution")):
        _validate_component(value, label)
    key = env.key()
    root = engine_cache_root() / f"{env.framework}-{env.name}-{selected}-{key[:_KEY_PREFIX]}"
    project = root / "project"
    venv = root / "venv"
    if root.is_symlink() or venv.is_symlink():
        msg = f"Worker environment cache path cannot be a symbolic link: {root}"
        raise ValueError(msg)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive(root.parent / f".{root.name}.lock"):
        _materialize_once(env.source, project, offline=offline)
        command = [uv_bin(), "sync", "--frozen", "--project", str(project)]
        if env.python is not None:
            command.extend(["--python", env.python])
        for extra in env.extras_for(selected):
            command.extend(["--extra", extra])
        if offline:
            command.append("--offline")
        child_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
        logger.info("Preparing %s %s worker environment (%s).", env.framework, env.name, selected)
        subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # Only fixed uv subcommands and validated project paths are passed.
            command,
            cwd=project,
            env=child_env,
            check=True,
            shell=False,
        )
        _write_marker(
            root,
            {
                "format_version": _MARKER_VERSION,
                "framework": env.framework,
                "name": env.name,
                "accelerator": selected,
                "key": key,
                "distribution": env.distribution,
                "source": env.source.provenance(),
            },
        )
    return PreparedEnvironment(env.framework, env.name, selected, key, env.distribution, root)


def list_prepared() -> list[PreparedEnvironment]:
    """List valid managed environments and their on-disk size.

    Returns:
        Environments sorted by framework, name, accelerator, and path.
    """
    prepared: list[PreparedEnvironment] = []
    for root, marker in _markers():
        prepared.append(
            PreparedEnvironment(
                framework=_marker_string(marker, "framework"),
                name=_marker_string(marker, "name"),
                accelerator=_marker_string(marker, "accelerator"),
                key=_marker_string(marker, "key"),
                distribution=_marker_string(marker, "distribution"),
                path=root,
                size_bytes=_directory_size(root),
            ),
        )
    return sorted(prepared, key=lambda item: (item.framework, item.name, item.accelerator, str(item.path)))


def prune_prepared(current: Iterable[WorkerEnv]) -> list[Path]:
    """Remove managed environments that no current engine or enablement declares.

    Args:
        current: Every environment declared by installed engines and their enablements.

    Returns:
        Paths of removed worker environments.
    """
    cache = engine_cache_root()
    if not cache.is_dir():
        return []

    current_keys = {(env.framework, env.name): env.key() for env in current}
    removed: list[Path] = []
    for root in cache.iterdir():
        if root.is_symlink() or not root.is_dir():
            continue
        marker_path = root / _MARKER
        if marker_path.is_symlink() or not marker_path.is_file():
            continue
        try:
            marker = _read_marker(root)
            identity = (_marker_string(marker, "framework"), _marker_string(marker, "name"))
            key = _marker_string(marker, "key")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            stale = True
        else:
            stale = current_keys.get(identity) != key
        if stale:
            shutil.rmtree(root)
            removed.append(root)
    return sorted(removed)


def prune() -> list[Path]:
    """Prune cached environments that installed engines and enablements no longer declare.

    Returns:
        Paths of removed worker environments.
    """
    from . import Engine  # ruff: ignore[import-outside-top-level]  # Discovery imports this module during package initialization.

    current: list[WorkerEnv] = []
    for name in _prepared_frameworks():
        try:
            current.extend(Engine.get(name).environments())
        except ValueError:
            continue
    return prune_prepared(current)


def _prepared_frameworks() -> set[str]:
    return {_marker_string(marker, "framework") for _, marker in _markers()}


def _markers() -> Iterator[tuple[Path, dict[str, Any]]]:
    cache = engine_cache_root()
    if not cache.is_dir():
        return
    for root in sorted(cache.iterdir()):
        if root.is_symlink() or not root.is_dir() or not (root / _MARKER).is_file():
            continue
        try:
            marker = _read_marker(root)
            for key in ("framework", "name", "accelerator", "key", "distribution"):
                _marker_string(marker, key)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            logger.warning(
                "Skipping worker environment %s with an outdated or invalid marker; "
                "`physicalai engine prune` removes it.",
                root,
            )
            continue
        yield root, marker


def interpreter_for(
    env: WorkerEnv,
    *,
    accelerator: str = "auto",
    interpreter: str | Path | None = None,
) -> Path:
    """Choose a supplied interpreter or provision the locked environment.

    Returns:
        The Python executable to run.
    """
    if interpreter is not None:
        return Path(interpreter).expanduser()
    return prepare(env, accelerator)


def scripts_dir(interpreter: str | Path) -> Path:
    """Return the scripts directory adjacent to a Python interpreter.

    Returns:
        Directory containing scripts installed beside the interpreter.
    """
    return Path(interpreter).expanduser().parent


def _python_path(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _materialize_once(source: ProjectSource, destination: Path, *, offline: bool) -> None:
    if destination.is_symlink():
        msg = f"Worker project cache path cannot be a symbolic link: {destination}"
        raise ValueError(msg)
    if destination.is_dir():
        return
    if destination.exists():
        msg = f"Worker project cache path is not a directory: {destination}"
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        source.materialize(staged, offline=offline)
        try:
            staged.rename(destination)
        except FileExistsError:
            if not destination.is_dir() or destination.is_symlink():
                raise


@contextlib.contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Serialize preparation of one environment across processes on POSIX hosts.

    Yields:
        Control while the lock is held.
    """
    try:
        import fcntl  # ruff: ignore[import-outside-top-level]  # POSIX-only module.
    except ImportError:
        yield
        return
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_marker(root: Path, marker: dict[str, Any]) -> None:
    marker_path = root / _MARKER
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as file:
        json.dump(marker, file, sort_keys=True)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(marker_path)


def _read_marker(root: Path) -> dict[str, Any]:
    marker_path = root / _MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        msg = f"Worker environment marker not found: {marker_path}"
        raise FileNotFoundError(msg)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if not isinstance(marker, dict):
        msg = f"Worker environment marker is not an object: {marker_path}"
        raise TypeError(msg)
    version = marker.get("format_version")
    if isinstance(version, bool) or version != _MARKER_VERSION:
        msg = f"Unsupported worker environment marker version {version!r}."
        raise ValueError(msg)
    return marker


def _marker_string(marker: object, key: str) -> str:
    if not isinstance(marker, dict) or not isinstance(marker.get(key), str):
        msg = f"Worker environment marker has an invalid {key!r} field."
        raise TypeError(msg)
    return marker[key]


def _directory_size(root: Path) -> int:
    size = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        size += path.stat().st_size
    return size


def _validate_component(value: str, label: str) -> None:
    if _COMPONENT_RE.fullmatch(value) is None:
        msg = f"Invalid worker environment {label}: {value!r}."
        raise ValueError(msg)


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


__all__ = [
    "PreparedEnvironment",
    "engine_cache_root",
    "interpreter_for",
    "list_prepared",
    "lock_hash",
    "locked_version",
    "prepare",
    "prepare_environment",
    "prune",
    "prune_prepared",
    "resolve_accelerator",
    "scripts_dir",
    "supported_versions",
    "uv_bin",
]
