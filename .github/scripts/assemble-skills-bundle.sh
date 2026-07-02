#!/usr/bin/env bash
# Build the physical-ai-studio/ tree for open-edge-platform/skills.
# Keep "include" copies in sync with skills/sync-manifest.yaml.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="${1:?usage: assemble-skills-bundle.sh <output-dir>}"
PRODUCT="$(grep '^product:' skills/sync-manifest.yaml | sed 's/^product:[[:space:]]*//')"
BUNDLE="$OUT/$PRODUCT"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE"

copy_skill_dirs() {
  local bucket_src="$1"
  local bucket_dest="$2"
  mkdir -p "$BUNDLE/$bucket_dest"
  shopt -s nullglob
  for dir in "$bucket_src"/*/; do
    [[ -f "$dir/SKILL.md" ]] || continue
    rsync -a --delete "$dir" "$BUNDLE/$bucket_dest/$(basename "$dir")/"
  done
  shopt -u nullglob
}

copy_skill_dirs skills/library library
copy_skill_dirs skills/application application

copy_file() {
  local src="$1"
  local dest="$2"
  if [[ ! -f "$src" ]]; then
    echo "Missing include source: $src" >&2
    exit 1
  fi
  mkdir -p "$BUNDLE/$(dirname "$dest")"
  cp "$src" "$BUNDLE/$dest"
}

copy_file skills/library/README.md library/README.md
copy_file skills/library/EVALUATION.md library/EVALUATION.md
copy_file skills/application/README.md application/README.md
copy_file skills/publish/oep/README.md README.md

manifest="$BUNDLE/manifest.yaml"
{
  echo "product: $PRODUCT"
  echo "source_repository: https://github.com/${GITHUB_REPOSITORY:-open-edge-platform/physical-ai-studio}"
  echo "source_ref: ${GITHUB_SHA:-local}"
  echo "source_ref_name: ${GITHUB_REF_NAME:-local}"
  echo "synced_at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "skills:"
  echo "  library:"
  for dir in skills/library/*/; do
    [[ -f "$dir/SKILL.md" ]] || continue
    echo "    - $(basename "$dir")"
  done
  echo "  application:"
  for dir in skills/application/*/; do
    [[ -f "$dir/SKILL.md" ]] || continue
    echo "    - $(basename "$dir")"
  done
} >"$manifest"

echo "Bundle written to $BUNDLE"
