#!/usr/bin/env python3
"""Create or verify .claude/skills and .agents/skills adapter symlinks."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def skill_dirs(root: Path) -> list[tuple[str, str]]:
    """Return (bucket, skill_name) for each canonical skill directory."""
    found: list[tuple[str, str]] = []
    for bucket in ("library", "application"):
        bucket_path = root / "skills" / bucket
        if not bucket_path.is_dir():
            continue
        for child in sorted(bucket_path.iterdir()):
            if child.is_dir() and (child / "SKILL.md").is_file():
                found.append((bucket, child.name))
    return found


def adapter_target(bucket: str, name: str) -> str:
    return f"../../skills/{bucket}/{name}"


def read_link(path: Path) -> str | None:
    if path.is_symlink():
        return os.readlink(path)
    return None


def remove_adapter(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir() and not path.is_symlink():
        raise RuntimeError(f"{path} is a real directory; remove it manually")


def create_adapter(link: Path, target: str) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    remove_adapter(link)
    abs_target = (link.parent / target).resolve()
    if not abs_target.is_dir():
        raise RuntimeError(f"canonical skill missing: {abs_target}")

    if platform.system() == "Windows":
        _create_windows_link(link, abs_target)
    else:
        link.symlink_to(target, target_is_directory=True)


def _create_windows_link(link: Path, abs_target: Path) -> None:
    try:
        link.symlink_to(abs_target, target_is_directory=True)
        return
    except OSError:
        pass

    # Directory junction does not require symlink privilege on many Windows setups.
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(abs_target)],
        check=True,
        capture_output=True,
        text=True,
    )


def sync_adapters(root: Path, check_only: bool) -> int:
    errors: list[str] = []
    adapters = (root / ".claude" / "skills", root / ".agents" / "skills")

    for bucket, name in skill_dirs(root):
        target = adapter_target(bucket, name)
        for adapter_root in adapters:
            link = adapter_root / name
            current = read_link(link)
            if current == target and link.exists():
                continue
            if check_only:
                errors.append(
                    f"{link}: expected symlink to {target!r}, got {current!r}"
                )
                continue
            create_adapter(link, target)

    if check_only and errors:
        for msg in errors:
            print(msg, file=sys.stderr)
        print(
            "Run: python3 .github/scripts/skills/link_skills.py",
            file=sys.stderr,
        )
        return 1

    if not check_only:
        names = sorted(
            {name for _, name in skill_dirs(root)},
        )
        for adapter_root in adapters:
            print(f"{adapter_root.relative_to(root)}:")
            for name in names:
                if (adapter_root / name).exists():
                    print(f"  {name}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if adapters are missing or out of date",
    )
    args = parser.parse_args()
    return sync_adapters(repo_root(), check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
