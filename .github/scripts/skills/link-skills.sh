#!/usr/bin/env bash
# Recreate flat .claude/skills and .agents/skills symlinks from skills/{library,application}/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p .claude/skills .agents/skills

link_skill() {
  local bucket="$1"
  local name="$2"
  local target="../../skills/$bucket/$name"
  ln -sf "$target" "$ROOT/.claude/skills/$name"
  ln -sf "$target" "$ROOT/.agents/skills/$name"
}

for bucket in library application; do
  bucket_dir="skills/$bucket"
  [[ -d "$bucket_dir" ]] || continue
  shopt -s nullglob
  for dir in "$bucket_dir"/*/; do
    [[ -f "$dir/SKILL.md" ]] || continue
    link_skill "$bucket" "$(basename "$dir")"
  done
  shopt -u nullglob
done

echo "Linked skills:"
ls -1 .claude/skills
