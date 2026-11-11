# PROGRESS

_Last updated: 2026-11-11 (Phase 3)_

## Done
- Phase 0: private GitHub repo, .gitignore, folder structure, first commit.

- Phase 1: RESEARCH.md written (resources, licenses, gap analysis, plan + estimates).

- Phase 2 (templates): `configs/{schema,lexicon,templates}.yaml`, `src/bolo/{config,schema,splits,generator,cli}.py`,
  `bolo generate` / `bolo stats`, 22 tests passing, `docs/DATASHEET.md`, sample at `samples/text_v0_sample.jsonl`.
  v0 = 5000/750/1000 rows, ~14% `none`, 3 scripts, generated in <1 s.

- Phase 3: normalizer (`normalize.py`), eval harness (`evaluate.py`), rule baseline, Bolo-tiny (PhonoHash, from
  scratch, 0.69M params), MuRIL joint tagger, Qwen2.5-0.5B LoRA (MLX 4-bit), environment grounding
  (`grounding.py`), Bolo-tiny ONNX int8 export (0.8 MB, no accuracy loss), LLM paraphrases (train + reviewed test).
  Results in `results/RESULTS.md`. 56 tests passing.

## In progress
- Waiting for go-ahead on Phase 4 (recording tool + synthetic speech + ASR notebooks).

## Next
- Phase 4: `bolo record`, TTS synthetic audio notebook (Kaggle), Whisper fine-tune notebook.
- Optional: regenerate paraphrases with an Apache-2.0 model (Qwen2.5-1.5B or 7B) for a releasable dataset.
- Tune the grounding threshold on real recordings.

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

- 2026-11-11: MuRIL's last layer has LayerNorm gain ~0.015 (near-constant output). Use the second-to-last layer + mean pool.
- 2026-11-11: mlx-lm completion datasets go through the chat template, so inference must apply it too.
- 2026-11-11: Paraphraser = mlx-community/Qwen2.5-3B-Instruct-4bit (Qwen Research License, research/eval only).
  Model deleted after use (as the user asked). para_* data and the `*_para*` Bolo-tiny models are research-only.
  The recommended release model is `phono_views_aug` (no paraphrases) until paraphrases are regenerated with an Apache-2.0 model.
- 2026-11-11: Paraphrase quality: 56% of test paraphrases kept after review; MuRIL agreed with 74% of train paraphrases.
- 2026-11-11: Negative results kept: consistency loss (−1.4 EM), MuRIL distillation (−4.0 EM).

## Open questions
- OpenSLR-104 license: OpenSLR says CC BY-SA 4.0, Srota card says CC BY 4.0. Treat as BY-SA.
- HiACC license on Zenodo: not yet confirmed.
- Qwen Research License: do its terms cover outputs (our paraphrases)? Until resolved, treat them as research-only.
