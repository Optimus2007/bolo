# PROGRESS

_Last updated: 2026-11-11 (Phase 2)_

## Done
- Phase 0: private GitHub repo, .gitignore, folder structure, first commit.

- Phase 1: RESEARCH.md written (resources, licenses, gap analysis, plan + estimates).

- Phase 2 (templates): `configs/{schema,lexicon,templates}.yaml`, `src/bolo/{config,schema,splits,generator,cli}.py`,
  `bolo generate` / `bolo stats`, 22 tests passing, `docs/DATASHEET.md`, sample at `samples/text_v0_sample.jsonl`.
  v0 = 5000/750/1000 rows, ~14% `none`, 3 scripts, generated in <1 s.

## In progress
- Phase 2b: local LLM paraphrasing (awaiting go-ahead; ~1.8 GB model download).

## Next
- Phase 3: rule/fuzzy baseline, Bolo-tiny (from scratch), MuRIL, Qwen-0.5B LoRA; eval + MLX export.

## How to resume
```
cd ~/Documents/bolo && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/bolo generate --out data/text/v0 --sample-out samples/text_v0_sample.jsonl
```

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

- 2026-11-11: Splits are by slot VALUE (stable SHA-1 hash, 70/15/15). ~20% of templates are test-only.
  Closed-vocab texts may also appear in train; they're flagged `meta.seen_in_train` rather than dropped
  (dropping starved train of short negatives).
- 2026-11-11: Each row carries `asr_text` (canonical Roman, the ASR target) and `tts_text` (mixed script,
  the TTS input), so the same generator feeds Phase 4.
- 2026-11-11: Per-intent quotas instead of random intent sampling (random sampling under-filled
  closed-vocab intents because of dedup).

## Open questions
- OpenSLR-104 license: OpenSLR says CC BY-SA 4.0, Srota card says CC BY 4.0. Treat as BY-SA.
- HiACC license on Zenodo: not yet confirmed.
