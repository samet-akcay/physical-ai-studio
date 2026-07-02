# Studio (application) agent skills

Skills for the GUI and orchestration stack under `application/`:

- `application/backend/` — FastAPI, job orchestration, OpenAPI
- `application/ui/` — React, generated API types
- `application/docker/` — compose, device setup

No application skills are checked in yet. Add them here with the `studio-` name prefix (see [`../README.md`](../README.md)).

When the first skill lands, add [`EVALUATION.md`](EVALUATION.md) with at least three scenarios per skill.

## Add a studio skill

```bash
NAME=studio-my-workflow
mkdir -p "skills/application/$NAME"
$EDITOR "skills/application/$NAME/SKILL.md"
python3 .github/scripts/skills/link_skills.py
```

Global authoring rules: [`../README.md`](../README.md).
