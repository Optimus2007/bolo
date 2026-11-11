# PROGRESS

_Last updated: 2026-11-11_

## Done
- Phase 0: private GitHub repo, .gitignore, folder structure, first commit.

## In progress
- Phase 1: research (RESEARCH.md).

## Next
- Present plan with time/disk/compute estimates; wait for approval before Phase 2.

## Decisions
- Layout: `src/bolo` (package), `configs/` (YAML schema), `scripts/`, `notebooks/` (Kaggle/Colab),
  `tests/`, `docs/`, `samples/` (small committed examples). `data/` and `models/` are gitignored.
- .gitignore blocks all audio formats, weights (safetensors/gguf/pt/npz/...), envs, and secrets.
- Licenses (proposed): Apache-2.0 for code, CC BY 4.0 for generated data.

## Open questions
- (none yet)
