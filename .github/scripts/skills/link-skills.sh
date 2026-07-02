#!/usr/bin/env bash
# Recreate flat .claude/skills and .agents/skills symlinks from skills/{library,application}/.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/link_skills.py" "$@"
