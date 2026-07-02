#!/usr/bin/env bash
# Validate skills/ layout, SKILL.md frontmatter, and client adapter symlinks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

errors=0

fail() {
  echo "::error::$1" >&2
  errors=$((errors + 1))
}

check_skill_dir() {
  local dir="$1"
  local name
  name="$(basename "$dir")"
  local skill_md="$dir/SKILL.md"

  if [[ ! -f "$skill_md" ]]; then
    fail "Missing SKILL.md in $dir"
    return
  fi

  local fm_name
  fm_name="$(sed -n '/^---$/,/^---$/p' "$skill_md" | sed -n 's/^name:[[:space:]]*//p' | head -1 | tr -d '\r')"
  if [[ -z "$fm_name" ]]; then
    fail "$skill_md: missing frontmatter name"
    return
  fi
  if [[ "$fm_name" != "$name" ]]; then
    fail "$skill_md: frontmatter name '$fm_name' must match directory '$name'"
  fi

  if find "$dir" -mindepth 1 -maxdepth 1 -type l | grep -q .; then
    fail "$dir: must not contain symlinks (canonical skill content only)"
  fi

  for adapter in .claude/skills .agents/skills; do
    local link="$adapter/$name"
    if [[ ! -L "$link" ]]; then
      fail "Missing symlink $link (run python3 .github/scripts/skills/link_skills.py)"
      continue
    fi
    if [[ ! -f "$link/SKILL.md" ]]; then
      fail "Broken symlink $link"
    fi
  done
}

for bucket in library application; do
  bucket_dir="skills/$bucket"
  if [[ ! -d "$bucket_dir" ]]; then
    fail "Missing $bucket_dir"
    continue
  fi
  shopt -s nullglob
  for dir in "$bucket_dir"/*/; do
    [[ -d "$dir" ]] || continue
    base="$(basename "$dir")"
    if [[ ! -f "$dir/SKILL.md" ]]; then
      continue
    fi
    check_skill_dir "$dir"
  done
  shopt -u nullglob
done

if [[ ! -f skills/sync-manifest.yaml ]]; then
  fail "Missing skills/sync-manifest.yaml"
fi

if [[ "$errors" -gt 0 ]]; then
  echo "Skills validation failed with $errors error(s)."
  exit 1
fi

echo "Skills validation passed."
