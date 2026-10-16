# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Declarative sources for worker projects.

A worker environment is built from a uv project (``pyproject.toml`` and
``uv.lock``). A source describes where that project comes from:

- :class:`BundledProject`: a project shipped inside a plugin, as LeRobot does;
- :class:`GitProject`: a framework repository's own project at an immutable
  commit, optionally with one or more patch series applied in order.

A patch series says where its patches live and how their content is pinned:

- :class:`BundledPatches`: patches stored in the plugin itself, the source of
  truth for their content. Their digests are computed from the files.
- :class:`PublishedPatches`: patches published by another repository at an
  immutable revision, pinned by per-file SHA-256 digests, optionally read
  from a local mirror.

Each source reports an :meth:`~ProjectSource.identity` that keys the cached
environment, so changing any pin or patch builds a new environment instead of
mutating an existing one. A series can also pin the Git tree its application
must produce, so every source of the same patches yields the same reviewed code.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # Fixed git argv lists; shell is disabled.
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from importlib.resources.abc import Traversable

logger = logging.getLogger(__name__)

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PATCH_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}")
_DOWNLOAD_TIMEOUT_SECONDS = 60
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "physicalai",
    "GIT_AUTHOR_EMAIL": "physicalai@localhost",
    "GIT_COMMITTER_NAME": "physicalai",
    "GIT_COMMITTER_EMAIL": "physicalai@localhost",
}


class ProjectSource(ABC):
    """Where a worker environment's uv project comes from."""

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return JSON-serializable inputs that determine the project's contents.

        Environment keys hash this value, so it must change exactly when the
        materialized project can change.
        """

    def provenance(self) -> dict[str, Any]:
        """Return where the project's inputs come from, for records and audits.

        Returns:
            The identity, unless a source records more detail than its content needs.
        """
        return self.identity()

    @abstractmethod
    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Write the project into the new directory *destination*."""

    def lock(self) -> bytes | None:  # noqa: PLR6301  # Overridden by sources that know their lock up front.
        """Return the lock before materialization, when the source carries one.

        Returns:
            ``uv.lock`` bytes, or ``None`` when the lock exists only after materialization.
        """
        return None


@dataclass(frozen=True)
class BundledProject(ProjectSource):
    """A uv project shipped inside a plugin distribution."""

    path: Path | Traversable

    def identity(self) -> dict[str, Any]:
        """Identify the project by its lock.

        Returns:
            The lock digest.
        """
        return {"kind": "bundled", "lock_sha256": hashlib.sha256(self.lock()).hexdigest()}

    def lock(self) -> bytes:
        """Read the bundled ``uv.lock``.

        Returns:
            Lock bytes.

        Raises:
            FileNotFoundError: If the project has no lock.
        """
        lock = self.path.joinpath("uv.lock")
        if not lock.is_file():
            msg = f"Bundled worker project {self.path} does not include uv.lock."
            raise FileNotFoundError(msg)
        return lock.read_bytes()

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Copy the bundled project."""
        del offline
        _copy_tree(self.path, destination)


@dataclass(frozen=True)
class PatchFile:
    """One file of a patch series and its expected SHA-256 digest."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        """Reject names that could escape the series directory and malformed digests.

        Raises:
            ValueError: If the name or digest is invalid.
        """
        _validate_patch_name(self.name)
        if _SHA256_RE.fullmatch(self.sha256) is None:
            msg = f"Patch {self.name!r} needs a lowercase hexadecimal SHA-256 digest."
            raise ValueError(msg)


@dataclass(frozen=True)
class PatchSeries(ABC):
    """An ordered series of ``git format-patch`` files applied on top of a Git project.

    Attributes:
        tree: Git tree the checkout must have after this series is applied, or
            ``None`` to skip the check.
        license: License of the patch files.
    """

    tree: str | None = field(default=None, kw_only=True)
    license: str | None = field(default=None, kw_only=True)

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return JSON-serializable inputs that determine the patched source."""

    @abstractmethod
    def patches(self) -> list[tuple[str, bytes]]:
        """Return ``(name, content)`` pairs in application order, verified where pinned."""


@dataclass(frozen=True)
class BundledPatches(PatchSeries):
    """Patches stored with the plugin, which owns and reviews their content.

    Use this when a team maintains its patches in this repository. Files apply
    in name order, which follows ``git format-patch`` numbering, unless *files*
    lists them explicitly. Digests are computed from the files, so nobody edits
    checksums. With a ``tree`` pin, the reviewed tree identifies the patched
    source and a changed patch fails until the pin is updated; without one, the
    file digests identify it.

    Attributes:
        directory: Directory holding the patch files.
        files: Optional explicit application order.
    """

    directory: Path | Traversable
    files: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Reject a malformed tree pin or explicit file name with ``ValueError``."""
        _validate_tree(self.tree)
        for name in self.files or ():
            _validate_patch_name(name)

    def identity(self) -> dict[str, Any]:
        """Identify the series by the digests of its files.

        Returns:
            File names and digests, tree, and license.
        """
        return {
            "kind": "bundled",
            "files": [{"name": name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in self.patches()],
            "tree": self.tree,
            "license": self.license,
        }

    def patches(self) -> list[tuple[str, bytes]]:
        """Read the patch files in application order.

        Returns:
            Patch names and contents.

        Raises:
            FileNotFoundError: If a listed file is missing or the directory holds no patches.
        """
        names = self.files
        if names is None:
            names = tuple(sorted(child.name for child in self.directory.iterdir() if child.name.endswith(".patch")))
        if not names:
            msg = f"No patch files found in {self.directory}."
            raise FileNotFoundError(msg)
        series = []
        for name in names:
            _validate_patch_name(name)
            path = self.directory.joinpath(name)
            if not path.is_file():
                msg = f"Patch file {name} is missing from {self.directory}."
                raise FileNotFoundError(msg)
            series.append((name, path.read_bytes()))
        return series


@dataclass(frozen=True)
class PublishedPatches(PatchSeries):
    """Patches published by another repository at an immutable revision.

    Use this while another team owns the patches. Each file is pinned by its
    SHA-256 digest and fetched from the pinned revision, or read from *mirror*
    when a local copy exists; both are verified identically.

    Attributes:
        repository: HTTPS GitHub repository that publishes the series.
        revision: Commit of *repository* that holds the reviewed files.
        directory: Directory of the series inside *repository*.
        files: Patch files in application order.
        mirror: Optional local copy of the files, for example for offline builds.

    A published series requires ``tree``: the files belong to another team, so
    the reviewed result is pinned explicitly.
    """

    repository: str
    revision: str
    directory: str
    files: tuple[PatchFile, ...]
    mirror: Path | Traversable | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        """Validate the immutable pins.

        Raises:
            ValueError: If a pin is malformed or the series is empty.
        """
        _github_slug(self.repository)
        if _COMMIT_RE.fullmatch(self.revision) is None:
            msg = "Published patch revision must be a full lowercase Git commit ID."
            raise ValueError(msg)
        _validate_tree(self.tree, required=True)
        if not self.files:
            msg = "A patch series needs at least one file."
            raise ValueError(msg)

    def identity(self) -> dict[str, Any]:
        """Return the pins that determine the patched source.

        Returns:
            Repository, revision, directory, files, tree, and license.
        """
        return {
            "kind": "published",
            "repository": self.repository,
            "revision": self.revision,
            "directory": self.directory,
            "files": [{"name": patch.name, "sha256": patch.sha256} for patch in self.files],
            "tree": self.tree,
            "license": self.license,
        }

    def patches(self) -> list[tuple[str, bytes]]:
        """Read and verify every file in application order.

        Returns:
            Patch names and verified contents.
        """
        return [(patch.name, self.read(patch)) for patch in self.files]

    def url(self, patch: PatchFile) -> str:
        """Return the immutable download URL of *patch*.

        Returns:
            Raw GitHub URL at the pinned revision.
        """
        return (
            f"https://raw.githubusercontent.com/{_github_slug(self.repository)}/{self.revision}/"
            f"{self.directory.strip('/')}/{patch.name}"
        )

    def read(self, patch: PatchFile) -> bytes:
        """Read one patch from the mirror or its pinned URL and verify it.

        Returns:
            The verified patch bytes.

        Raises:
            ValueError: If the bytes do not match the pinned digest.
        """
        local = self.mirror.joinpath(patch.name) if self.mirror is not None else None
        data = local.read_bytes() if local is not None and local.is_file() else _download(self.url(patch))
        digest = hashlib.sha256(data).hexdigest()
        if digest != patch.sha256:
            msg = f"SHA-256 mismatch for patch {patch.name}: expected {patch.sha256}, got {digest}."
            raise ValueError(msg)
        return data


@dataclass(frozen=True)
class GitProject(ProjectSource):
    """A framework repository's own uv project at an immutable commit.

    Attributes:
        repository: HTTPS Git repository of the framework.
        revision: Full commit ID to check out.
        patches: Optional patch series, or several applied in order. Layers let
            a plugin keep another team's published series and add its own, or
            move patches into the plugin one series at a time.
    """

    repository: str
    revision: str
    patches: PatchSeries | tuple[PatchSeries, ...] | None = None

    @property
    def layers(self) -> tuple[PatchSeries, ...]:
        """Patch series in application order."""
        if self.patches is None:
            return ()
        if isinstance(self.patches, PatchSeries):
            return (self.patches,)
        return tuple(self.patches)

    def __post_init__(self) -> None:
        """Validate the repository URL and commit.

        Raises:
            ValueError: If the repository is not HTTPS or the revision is not a full commit ID.
        """
        if urlparse(self.repository).scheme != "https":
            msg = f"Git projects must use HTTPS repositories, got {self.repository!r}."
            raise ValueError(msg)
        if _COMMIT_RE.fullmatch(self.revision) is None:
            msg = f"Git project revision must be a full lowercase commit ID, got {self.revision!r}."
            raise ValueError(msg)

    def identity(self) -> dict[str, Any]:
        """Identify the checkout by content.

        A commit ID identifies the upstream source. When the last patch layer
        pins its resulting tree, that tree identifies the patched source no
        matter where the patches come from, so moving the same patches from
        another repository into a plugin keeps the environment, and packages
        built in it stay valid. Without a final tree pin, every layer's patch
        digests identify the source.

        Returns:
            Commit plus final tree, or commit plus every layer's identity.
        """
        layers = self.layers
        if layers and layers[-1].tree is not None:
            return {"kind": "git", "revision": self.revision, "tree": layers[-1].tree}
        return {"kind": "git", "revision": self.revision, "patches": [layer.identity() for layer in layers]}

    def provenance(self) -> dict[str, Any]:
        """Record the repository and where every patch layer comes from.

        Returns:
            Repository, commit, and each layer's pins.
        """
        return {
            "kind": "git",
            "repository": self.repository,
            "revision": self.revision,
            "patches": [layer.identity() for layer in self.layers],
        }

    def materialize(self, destination: Path, *, offline: bool = False) -> None:
        """Check out the pinned commit and apply the verified patch series.

        Raises:
            RuntimeError: If offline, or if the checkout or patched tree differs from its pin.
        """
        if offline:
            msg = f"Cannot fetch {self.repository} offline; prepare the environment with network access first."
            raise RuntimeError(msg)
        destination.mkdir(parents=True)
        _git(["init", "--quiet"], cwd=destination)
        _git(["remote", "add", "origin", self.repository], cwd=destination)
        logger.info("Fetching %s at %s.", self.repository, self.revision)
        _git(["fetch", "--quiet", "--depth", "1", "origin", self.revision], cwd=destination)
        _git(["-c", "advice.detachedHead=false", "checkout", "--quiet", "FETCH_HEAD"], cwd=destination)
        head = _git(["rev-parse", "HEAD"], cwd=destination, capture=True)
        if head != self.revision:
            msg = f"Fetched commit {head} does not match the pinned commit {self.revision}."
            raise RuntimeError(msg)
        for layer in self.layers:
            _apply(layer, destination)


def _apply(series: PatchSeries, checkout: Path) -> None:
    patches = series.patches()
    with tempfile.TemporaryDirectory(prefix="physicalai-patches-") as temporary:
        files = []
        for name, data in patches:
            path = Path(temporary) / name
            path.write_bytes(data)
            files.append(str(path))
        logger.info("Applying %d %s patches.", len(files), series.identity()["kind"])
        _git(
            [
                "-c",
                "commit.gpgsign=false",
                "am",
                "--quiet",
                "--whitespace=fix",
                "--committer-date-is-author-date",
                *files,
            ],
            cwd=checkout,
            extra_env=_GIT_IDENTITY,
        )
    if series.tree is None:
        return
    tree = _git(["rev-parse", "HEAD^{tree}"], cwd=checkout, capture=True)
    if tree != series.tree:
        msg = f"Patched source tree {tree} does not match the reviewed tree {series.tree}."
        raise RuntimeError(msg)


def _validate_patch_name(name: str) -> None:
    if _PATCH_NAME_RE.fullmatch(name) is None or name in {".", ".."}:
        msg = f"Invalid patch file name {name!r}."
        raise ValueError(msg)


def _validate_tree(tree: str | None, *, required: bool = False) -> None:
    if tree is None and not required:
        return
    if tree is None or _COMMIT_RE.fullmatch(tree) is None:
        msg = "A patch series tree must be a full lowercase Git object ID."
        raise ValueError(msg)


def _github_slug(repository: str) -> str:
    parsed = urlparse(repository)
    if parsed.scheme != "https" or parsed.netloc != "github.com":
        msg = f"Patch series must be published in an HTTPS GitHub repository, got {repository!r}."
        raise ValueError(msg)
    return parsed.path.strip("/").removesuffix(".git")


def _download(url: str) -> bytes:
    if not url.startswith("https://raw.githubusercontent.com/"):
        msg = f"Refusing to download from an unpinned location: {url}"
        raise ValueError(msg)
    logger.info("Downloading %s.", url)
    # nosec B310 - only HTTPS raw.githubusercontent.com URLs at pinned revisions reach this call.
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310  # nosec B310
        return response.read()


def _git(
    arguments: Sequence[str],
    *,
    cwd: Path,
    capture: bool = False,
    extra_env: Mapping[str, str] | None = None,
) -> str:
    executable = shutil.which("git")
    if executable is None:
        msg = "Git project sources require git."
        raise RuntimeError(msg)
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # argv list; shell disabled.
        [executable, *arguments],
        cwd=cwd,
        env={**os.environ, **(extra_env or {})},
        check=True,
        shell=False,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def _copy_tree(source: Path | Traversable, destination: Path) -> None:
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_tree(child, destination / child.name)
        return
    if not source.is_file():
        msg = f"Worker project resource is not a file or directory: {source}"
        raise FileNotFoundError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


__all__ = [
    "BundledPatches",
    "BundledProject",
    "GitProject",
    "PatchFile",
    "PatchSeries",
    "ProjectSource",
    "PublishedPatches",
]
