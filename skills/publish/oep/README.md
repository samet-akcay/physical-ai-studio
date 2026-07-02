# Physical AI Studio skills

Agent skills for [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio): training policies, datasets, benchmarking, and export in `library/`, plus GUI/orchestration skills under `application/` when published.

**Do not edit this tree by hand.** It is synced from the Studio repository. Change skills in `physical-ai-studio` under `skills/` and merge to `main`; CI opens a PR here.

## Layout

- `library/` — `physicalai-train` workflows (`library-*` skills)
- `application/` — `application/backend`, `ui`, `docker` workflows (`studio-*` skills)
- `manifest.yaml` — source commit and skill list for this sync

## Install into a project

Symlink or copy individual skill folders into your agent skills path (flat adapter layout):

```bash
# Example: one library skill
ln -sf "/path/to/this/repo/physical-ai-studio/library/library-training-a-policy" \
  .agents/skills/library-training-a-policy
ln -sf "/path/to/this/repo/physical-ai-studio/library/library-training-a-policy" \
  .claude/skills/library-training-a-policy
```

Use `.agents/skills/` and/or `.claude/skills/` depending on your client. See Agent Skills: https://agentskills.io/

## Source

See `manifest.yaml` for the exact `physical-ai-studio` git SHA that produced this bundle.
