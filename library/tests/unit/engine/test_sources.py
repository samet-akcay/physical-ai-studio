# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Project source and patch series tests."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from physicalai.engine import BundledPatches, BundledProject, GitProject, PatchFile, PublishedPatches
from physicalai.engine import _sources

_SUITE = "https://github.com/open-edge-platform/robotics-ai-suite.git"
_COMMIT = "2960959ae6cc6d45150316ed6612f686c2d915b7"


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, Path, str]:
    """Create a repository with one commit and a patch that changes it.

    Returns:
        The checkout at the base commit, the patch file, and the patched tree.
    """
    if shutil.which("git") is None:
        pytest.skip("git is required")
    repository = tmp_path / "upstream"
    repository.mkdir()
    _git("init", "--quiet", cwd=repository)
    (repository / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git("add", "model.py", cwd=repository)
    _git("commit", "--quiet", "-m", "base", cwd=repository)
    (repository / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git("commit", "--quiet", "-am", "enable", cwd=repository)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=repository)
    patch = Path(_git("format-patch", "-1", "HEAD", "-o", str(tmp_path / "patches"), cwd=repository))
    _git("reset", "--quiet", "--hard", "HEAD~1", cwd=repository)
    return repository, patch, tree


def _series(patch: Path, tree: str, *, digest: str | None = None, mirror: Path | None = None) -> PublishedPatches:
    return PublishedPatches(
        repository=_SUITE,
        revision=_COMMIT,
        directory="src/pipelines/demo/patches",
        files=(PatchFile(patch.name, digest or hashlib.sha256(patch.read_bytes()).hexdigest()),),
        tree=tree,
        license="Apache-2.0",
        mirror=mirror,
    )


def test_mirrored_published_patches_are_verified_applied_and_checked_against_the_tree(
    upstream: tuple[Path, Path, str],
) -> None:
    repository, patch, tree = upstream
    _sources._apply(_series(patch, tree, mirror=patch.parent), repository)

    assert (repository / "model.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert _git("rev-parse", "HEAD^{tree}", cwd=repository) == tree


def test_patched_trees_that_differ_from_the_review_are_rejected(upstream: tuple[Path, Path, str]) -> None:
    repository, patch, _ = upstream
    with pytest.raises(RuntimeError, match="does not match the reviewed tree"):
        _sources._apply(_series(patch, "0" * 40, mirror=patch.parent), repository)


def test_tampered_patches_are_rejected_before_application(upstream: tuple[Path, Path, str]) -> None:
    repository, patch, tree = upstream
    series = _series(patch, tree, digest="0" * 64, mirror=patch.parent)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _sources._apply(series, repository)
    assert (repository / "model.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_missing_mirror_files_fall_back_to_the_pinned_url(
    monkeypatch: pytest.MonkeyPatch,
    upstream: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    _, patch, tree = upstream
    urls: list[str] = []

    def download(url: str) -> bytes:
        urls.append(url)
        return patch.read_bytes()

    monkeypatch.setattr(_sources, "_download", download)
    series = _series(patch, tree, mirror=tmp_path / "empty")

    assert series.read(series.files[0]) == patch.read_bytes()
    assert urls == [
        f"https://raw.githubusercontent.com/open-edge-platform/robotics-ai-suite/{_COMMIT}/src/pipelines/demo/patches/{patch.name}"
    ]


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: PatchFile("../escape.patch", "0" * 64), "Invalid patch file name"),
        (lambda: PatchFile("0001.patch", "abc"), "SHA-256"),
        (lambda: GitProject("http://github.com/x/y.git", "a" * 40), "HTTPS"),
        (lambda: GitProject("https://github.com/x/y.git", "main"), "full lowercase commit"),
        (
            lambda: PublishedPatches(
                "https://gitlab.com/x/y.git", _COMMIT, "p", (PatchFile("a.patch", "0" * 64),), tree="0" * 40
            ),
            "GitHub",
        ),
        (lambda: PublishedPatches(_SUITE, _COMMIT, "p", (), tree="0" * 40), "at least one file"),
        (lambda: PublishedPatches(_SUITE, _COMMIT, "p", (PatchFile("a.patch", "0" * 64),)), "tree must be"),
        (lambda: BundledPatches(Path("p"), tree="main"), "full lowercase Git object ID"),
        (lambda: BundledPatches(Path("p"), files=("../escape.patch",)), "Invalid patch file name"),
    ],
)
def test_pins_must_be_immutable_and_well_formed(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()  # type: ignore[operator]


def test_downloads_are_limited_to_pinned_raw_github_urls() -> None:
    with pytest.raises(ValueError, match="unpinned location"):
        _sources._download("https://example.com/0001.patch")


def test_git_projects_identify_by_commit_and_patches_and_refuse_offline(tmp_path: Path) -> None:
    patch = tmp_path / "0001.patch"
    patch.write_text("x", encoding="utf-8")
    plain = GitProject("https://github.com/NVIDIA/Isaac-GR00T.git", "a" * 40)
    patched = GitProject(plain.repository, plain.revision, _series(patch, "b" * 40))

    assert plain.identity() == {"kind": "git", "revision": "a" * 40, "patches": []}
    assert patched.identity() == {"kind": "git", "revision": "a" * 40, "tree": "b" * 40}
    assert patched.provenance()["patches"][0]["repository"] == _SUITE
    assert plain.lock() is None
    with pytest.raises(RuntimeError, match="offline"):
        plain.materialize(tmp_path / "checkout", offline=True)


def test_bundled_projects_identify_by_their_lock(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (project / "pyproject.toml").write_text("[project]\nname='w'\n", encoding="utf-8")
    source = BundledProject(project)

    assert source.identity() == {"kind": "bundled", "lock_sha256": hashlib.sha256(b"version = 1\n").hexdigest()}
    source.materialize(tmp_path / "copy")
    assert (tmp_path / "copy" / "uv.lock").read_text(encoding="utf-8") == "version = 1\n"


def _second_patch(repository: Path, directory: Path) -> tuple[Path, str]:
    """Create a patch on top of the first one and return it with its resulting tree.

    Returns:
        The patch file and the tree after applying both patches.
    """
    (repository / "model.py").write_text("VALUE = 2\nENABLED = True\n", encoding="utf-8")
    _git("commit", "--quiet", "-am", "enable more", cwd=repository)
    tree = _git("rev-parse", "HEAD^{tree}", cwd=repository)
    patch = Path(_git("format-patch", "-1", "HEAD", "--start-number", "2", "-o", str(directory), cwd=repository))
    return patch, tree


def test_bundled_patches_apply_in_name_order_without_external_pins(
    upstream: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    repository, patch, tree = upstream
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / patch.name).write_bytes(patch.read_bytes())
    _git("am", "--quiet", str(patch), cwd=repository)
    second, final_tree = _second_patch(repository, bundled)
    _git("reset", "--quiet", "--hard", "HEAD~2", cwd=repository)
    (bundled / "README.md").write_text("not a patch", encoding="utf-8")

    series = BundledPatches(bundled, tree=final_tree, license="Apache-2.0")
    _sources._apply(series, repository)

    assert [name for name, _ in series.patches()] == [patch.name, second.name]
    assert (repository / "model.py").read_text(encoding="utf-8") == "VALUE = 2\nENABLED = True\n"
    assert _git("rev-parse", "HEAD^{tree}", cwd=repository) == final_tree


def test_bundled_patch_identity_follows_file_content(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "0001-a.patch").write_text("one", encoding="utf-8")
    series = BundledPatches(bundled)
    before = series.identity()

    (bundled / "0001-a.patch").write_text("two", encoding="utf-8")

    assert before["kind"] == "bundled"
    assert before["files"][0]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert series.identity() != before
    assert BundledPatches(bundled, files=("0001-a.patch",)).identity() == series.identity()
    with pytest.raises(FileNotFoundError, match="missing"):
        BundledPatches(bundled, files=("0002-b.patch",)).patches()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No patch files"):
        BundledPatches(empty).patches()


def test_bundled_patches_without_a_tree_pin_skip_the_tree_check(
    upstream: tuple[Path, Path, str], tmp_path: Path
) -> None:
    repository, patch, tree = upstream
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / patch.name).write_bytes(patch.read_bytes())

    _sources._apply(BundledPatches(bundled), repository)

    assert _git("rev-parse", "HEAD^{tree}", cwd=repository) == tree


def test_layers_apply_published_then_bundled_patches_with_per_layer_trees(
    upstream: tuple[Path, Path, str],
    tmp_path: Path,
) -> None:
    repository, patch, tree = upstream
    _git("am", "--quiet", str(patch), cwd=repository)
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    _, final_tree = _second_patch(repository, bundled)
    _git("reset", "--quiet", "--hard", "HEAD~2", cwd=repository)
    project = GitProject(
        "https://github.com/NVIDIA/Isaac-GR00T.git",
        "a" * 40,
        patches=(_series(patch, tree, mirror=patch.parent), BundledPatches(bundled, tree=final_tree)),
    )

    for layer in project.layers:
        _sources._apply(layer, repository)

    assert [layer["kind"] for layer in project.provenance()["patches"]] == ["published", "bundled"]
    assert project.identity() == {"kind": "git", "revision": "a" * 40, "tree": final_tree}
    assert _git("rev-parse", "HEAD^{tree}", cwd=repository) == final_tree


def test_moving_published_patches_into_a_plugin_keeps_the_environment_key(tmp_path: Path) -> None:
    patch = tmp_path / "0001.patch"
    patch.write_text("x", encoding="utf-8")
    final = "c" * 40
    published = GitProject("https://github.com/NVIDIA/Isaac-GR00T.git", "a" * 40, _series(patch, final))
    bundled = GitProject(published.repository, published.revision, BundledPatches(tmp_path, tree=final))
    layered = GitProject(
        published.repository,
        published.revision,
        (_series(patch, "b" * 40), BundledPatches(tmp_path, tree=final)),
    )
    other_result = GitProject(published.repository, published.revision, BundledPatches(tmp_path, tree="d" * 40))
    unpinned = GitProject(published.repository, published.revision, BundledPatches(tmp_path))

    assert published.identity() == bundled.identity() == layered.identity()
    assert published.provenance() != bundled.provenance()
    assert other_result.identity() != bundled.identity()
    assert unpinned.identity()["patches"][0]["kind"] == "bundled"
