# PROGRESS

_Last updated: 2026-11-11_

## Done
- Phase 0: private GitHub repo, .gitignore, folder structure, first commit.

- Phase 1: RESEARCH.md written (resources, licenses, gap analysis, plan + estimates).

## In progress
- WAITING for approval of the Phase 2 plan.

## Next
- Phase 2: YAML schema, template generator, variant tables, unseen-slot split, datasheet, tests.

## Decisions
- Layout: `src/bolo` (package), `configs/` (YAML schema), `scripts/`, `notebooks/` (Kaggle/Colab),
  `tests/`, `docs/`, `samples/` (small committed examples). `data/` and `models/` are gitignored.
- .gitignore blocks all audio formats, weights (safetensors/gguf/pt/npz/...), envs, and secrets.
- Licenses (proposed): Apache-2.0 for code, CC BY 4.0 for generated data.

- Phase 1 conclusions: closest prior work is Hinglish-TOP (text, Apache-2.0, no desktop domain).
  ASR fine-tune starting point: Oriserve Whisper-Hindi2Hinglish-Swift (whisper-base, Apache-2.0, Roman output).
  Text models to compare: MuRIL joint tagger vs Qwen 0.5B/0.6B LoRA (+ rule/fuzzy baseline).
  TTS teacher: Indic Parler-TTS (Apache-2.0, gated), run on Kaggle.
- lipi-hn is CC BY-NC-SA: use it as an eval baseline only, never build on its weights.

- 2026-11-11: Direction = Bolo-Edge (see RESEARCH.md §6). Edge targets: M2 Mac + Android phone.
  Add Bolo-tiny (from-scratch tiny SLU model) to Phase 3; direct speech→intent as a stretch goal.
- 2026-11-11: Zero-cost project. No paid APIs. Paraphrasing = local MLX model, after templates-first.
  Heavy compute = free Kaggle/Colab.

## Open questions
- OpenSLR-104 license: OpenSLR says CC BY-SA 4.0, Srota card says CC BY 4.0. Treat as BY-SA.
- HiACC license on Zenodo: not yet confirmed.
